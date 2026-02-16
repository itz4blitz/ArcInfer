/// ArcInfer Solana Program
///
/// This Anchor program is the on-chain entrypoint for encrypted sentiment
/// classification. It handles:
///
/// 1. Initialization: Register the encrypted computation definition
/// 2. Classification: Accept encrypted embeddings, queue MPC computation
/// 3. Callback: Receive and verify MPC results, emit events
///
/// The actual classification runs inside the Arcium MPC network.
/// This program just orchestrates the lifecycle.
use anchor_lang::prelude::*;
use arcium_anchor::prelude::*;
use arcium_client::idl::arcium::types::{
    CallbackAccount, CircuitSource, OffChainCircuitSource, OnChainCircuitSource,
};
use arcium_macros::circuit_hash;

const COMP_DEF_OFFSET_CLASSIFY: u32 = comp_def_offset("classify");
const COMP_DEF_OFFSET_CLASSIFY_REVEAL: u32 = comp_def_offset("classify_reveal");

declare_id!("2UEesrBiknFE3BoAh5BtZwbr5y2AFvWe2wksVi3MqeX9");

const CLASSIFY_RESULT_SEED: &[u8] = b"classify_result";

#[account]
pub struct ClassificationResult {
    pub computation_account: Pubkey,
    pub is_set: bool,
    pub class: u8,
}

impl ClassificationResult {
    // 32 + 1 + 1
    pub const SIZE: usize = 34;
}

#[arcium_program]
pub mod arcinfer {
    use super::*;

    // =========================================================================
    // Initialization
    // =========================================================================

    pub fn init_classify_comp_def(ctx: Context<InitClassifyCompDef>) -> Result<()> {
        let circuit_source = match option_env!("CLASSIFY_CIRCUIT_URL") {
            Some(url) => Some(CircuitSource::OffChain(OffChainCircuitSource {
                source: url.to_string(),
                hash: circuit_hash!("classify"),
            })),
            None => Some(CircuitSource::OnChain(OnChainCircuitSource {
                is_completed: false,
                upload_auth: ctx.accounts.payer.key(),
            })),
        };
        init_comp_def(ctx.accounts, circuit_source, None)?;
        Ok(())
    }

    pub fn init_classify_reveal_comp_def(ctx: Context<InitClassifyRevealCompDef>) -> Result<()> {
        let circuit_source = match option_env!("CLASSIFY_REVEAL_CIRCUIT_URL") {
            Some(url) => Some(CircuitSource::OffChain(OffChainCircuitSource {
                source: url.to_string(),
                hash: circuit_hash!("classify_reveal"),
            })),
            None => Some(CircuitSource::OnChain(OnChainCircuitSource {
                is_completed: false,
                upload_auth: ctx.accounts.payer.key(),
            })),
        };
        init_comp_def(ctx.accounts, circuit_source, None)?;
        Ok(())
    }

    // =========================================================================
    // Classification
    // =========================================================================

    /// Submit encrypted embedding for classification.
    /// Uses Vec<[u8; 32]> instead of [[u8; 32]; 16] to avoid Solana stack overflow.
    pub fn classify(
        ctx: Context<Classify>,
        computation_offset: u64,
        encrypted_features: Vec<[u8; 32]>,
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        require!(
            encrypted_features.len() == 16,
            ErrorCode::InvalidFeatureCount
        );

        let mut args = ArgBuilder::new()
            .x25519_pubkey(pub_key)
            .plaintext_u128(nonce);

        for i in 0..16 {
            args = args.encrypted_i32(encrypted_features[i]);
        }

        let args = args.build();

        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            vec![ClassifyCallback::callback_ix(
                computation_offset,
                &ctx.accounts.mxe_account,
                &[],
            )?],
            1,
            0,
        )?;

        Ok(())
    }

    /// Submit embedding for classification with revealed (plaintext) result.
    pub fn classify_reveal(
        ctx: Context<ClassifyReveal>,
        computation_offset: u64,
        encrypted_features: Vec<[u8; 32]>,
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        require!(
            encrypted_features.len() == 16,
            ErrorCode::InvalidFeatureCount
        );

        let mut args = ArgBuilder::new()
            .x25519_pubkey(pub_key)
            .plaintext_u128(nonce);

        for i in 0..16 {
            args = args.encrypted_i32(encrypted_features[i]);
        }

        let args = args.build();

        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        // Initialize/clear the result PDA so the client can poll it reliably.
        ctx.accounts.result_account.computation_account = ctx.accounts.computation_account.key();
        ctx.accounts.result_account.is_set = false;
        ctx.accounts.result_account.class = 0;

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            vec![ClassifyRevealCallback::callback_ix(
                computation_offset,
                &ctx.accounts.mxe_account,
                &[CallbackAccount {
                    pubkey: ctx.accounts.result_account.key(),
                    is_writable: true,
                }],
            )?],
            1,
            0,
        )?;

        Ok(())
    }

    // =========================================================================
    // Callbacks (account structs auto-generated by #[arcium_callback])
    // =========================================================================

    #[arcium_callback(encrypted_ix = "classify")]
    pub fn classify_callback(
        ctx: Context<ClassifyCallback>,
        output: SignedComputationOutputs<ClassifyOutput>,
    ) -> Result<()> {
        match output.verify_output(
            &ctx.accounts.cluster_account,
            &ctx.accounts.computation_account,
        ) {
            Ok(_result) => {
                emit!(ClassificationCompleteEvent {
                    computation_offset: ctx.accounts.computation_account.key(),
                    success: true,
                });
            }
            Err(_) => {
                emit!(ClassificationCompleteEvent {
                    computation_offset: ctx.accounts.computation_account.key(),
                    success: false,
                });
                return Err(ErrorCode::AbortedComputation.into());
            }
        }

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "classify_reveal")]
    pub fn classify_reveal_callback(
        ctx: Context<ClassifyRevealCallback>,
        output: SignedComputationOutputs<ClassifyRevealOutput>,
    ) -> Result<()> {
        match output.verify_output(
            &ctx.accounts.cluster_account,
            &ctx.accounts.computation_account,
        ) {
            Ok(ClassifyRevealOutput { field_0: class }) => {
                ctx.accounts.result_account.computation_account =
                    ctx.accounts.computation_account.key();
                ctx.accounts.result_account.is_set = true;
                ctx.accounts.result_account.class = class;
                emit!(ClassificationRevealedEvent {
                    computation_offset: ctx.accounts.computation_account.key(),
                    class,
                });
            }
            Err(_) => {
                return Err(ErrorCode::AbortedComputation.into());
            }
        }

        Ok(())
    }
}

// =========================================================================
// Account Structs — Init Computation Definitions
// =========================================================================

#[init_computation_definition_accounts("classify", payer)]
#[derive(Accounts)]
pub struct InitClassifyCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    #[account(mut)]
    /// CHECK: comp_def_account is validated by the Arcium program CPI.
    pub comp_def_account: UncheckedAccount<'info>,
    #[account(mut, address = derive_mxe_lut_pda!(mxe_account.lut_offset_slot))]
    /// CHECK: address_lookup_table is validated by the Arcium program CPI.
    pub address_lookup_table: UncheckedAccount<'info>,
    #[account(address = LUT_PROGRAM_ID)]
    /// CHECK: lut_program is the Address Lookup Table program.
    pub lut_program: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("classify_reveal", payer)]
#[derive(Accounts)]
pub struct InitClassifyRevealCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    #[account(mut)]
    /// CHECK: comp_def_account is validated by the Arcium program CPI.
    pub comp_def_account: UncheckedAccount<'info>,
    #[account(mut, address = derive_mxe_lut_pda!(mxe_account.lut_offset_slot))]
    /// CHECK: address_lookup_table is validated by the Arcium program CPI.
    pub address_lookup_table: UncheckedAccount<'info>,
    #[account(address = LUT_PROGRAM_ID)]
    /// CHECK: lut_program is the Address Lookup Table program.
    pub lut_program: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

// =========================================================================
// Account Structs — Queue Computation
// =========================================================================

#[queue_computation_accounts("classify", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct Classify<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!(),
    )]
    pub sign_pda_account: Account<'info, ArciumSignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    #[account(mut, address = derive_mempool_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: mempool_account is validated by the Arcium program CPI.
    pub mempool_account: UncheckedAccount<'info>,
    #[account(mut, address = derive_execpool_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: executing_pool is validated by the Arcium program CPI.
    pub executing_pool: UncheckedAccount<'info>,
    #[account(mut, address = derive_comp_pda!(computation_offset, mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: computation_account is validated by the Arcium program CPI.
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CLASSIFY))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(mut, address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[queue_computation_accounts("classify_reveal", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct ClassifyReveal<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!(),
    )]
    pub sign_pda_account: Account<'info, ArciumSignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    #[account(mut, address = derive_mempool_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: mempool_account is validated by the Arcium program CPI.
    pub mempool_account: UncheckedAccount<'info>,
    #[account(mut, address = derive_execpool_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: executing_pool is validated by the Arcium program CPI.
    pub executing_pool: UncheckedAccount<'info>,
    #[account(mut, address = derive_comp_pda!(computation_offset, mxe_account, ErrorCode::ClusterNotSet))]
    /// CHECK: computation_account is validated by the Arcium program CPI.
    pub computation_account: UncheckedAccount<'info>,
    // Result PDA: init_if_needed so the same computation_offset can be retried
    // without "already initialized" errors. Fields are cleared to is_set=false
    // before queue_computation (see classify_reveal instruction body).
    // The PDA must exist before the callback executes — Arcium cannot create
    // accounts during callbacks.
    #[account(
        init_if_needed,
        payer = payer,
        space = 8 + ClassificationResult::SIZE,
        seeds = [CLASSIFY_RESULT_SEED, computation_account.key().as_ref()],
        bump,
    )]
    pub result_account: Account<'info, ClassificationResult>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CLASSIFY_REVEAL))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(mut, address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

// =========================================================================
// Account Structs — Callbacks
// =========================================================================

#[callback_accounts("classify")]
#[derive(Accounts)]
pub struct ClassifyCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CLASSIFY))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: computation_account is validated by the Arcium callback verification.
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_cluster_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    pub cluster_account: Account<'info, Cluster>,
    /// CHECK: instructions_sysvar is the Instructions sysvar.
    #[account(address = anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

/// SAFETY: Custom callback accounts MUST come after the 6 standard Arcium
/// callback accounts. The required ordering is:
///
///   1. arcium_program
///   2. comp_def_account
///   3. mxe_account
///   4. computation_account
///   5. cluster_account
///   6. instructions_sysvar
///   7+ custom accounts (result_account here)
///
/// If this ordering is violated, Arcium will not include the custom accounts
/// in the callback transaction. The callback will silently skip writing to the
/// result PDA — `callbackTransactionsSubmittedBm` stays 0 and `is_set` never
/// flips to true.
///
/// Regression test: tests/arcinfer.ts "classifies and reveals sentiment via MPC"
/// will timeout if this ordering breaks.
#[callback_accounts("classify_reveal")]
#[derive(Accounts)]
pub struct ClassifyRevealCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CLASSIFY_REVEAL))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: computation_account is validated by the Arcium callback verification.
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_cluster_pda!(mxe_account, ErrorCode::ClusterNotSet))]
    pub cluster_account: Account<'info, Cluster>,
    /// CHECK: instructions_sysvar is the Instructions sysvar.
    #[account(address = anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,

    // --- Custom callback accounts below (position 7+) ---
    // These are appended by the Arcium callback builder via CallbackAccount.
    // Moving result_account above instructions_sysvar will break the callback.
    #[account(
        mut,
        seeds = [CLASSIFY_RESULT_SEED, computation_account.key().as_ref()],
        bump,
    )]
    pub result_account: Account<'info, ClassificationResult>,
}

// =========================================================================
// Events
// =========================================================================

#[event]
pub struct ClassificationCompleteEvent {
    pub computation_offset: Pubkey,
    pub success: bool,
}

#[event]
pub struct ClassificationRevealedEvent {
    pub computation_offset: Pubkey,
    pub class: u8,
}

// =========================================================================
// Error codes
// =========================================================================

#[error_code]
pub enum ErrorCode {
    #[msg("MPC computation was aborted or produced invalid output")]
    AbortedComputation,
    #[msg("Cluster not set in MXE account")]
    ClusterNotSet,
    #[msg("Expected exactly 16 encrypted features")]
    InvalidFeatureCount,
}
