# Arcium Examples Reference - Complete Source Code

Source: https://github.com/arcium-hq/examples

All examples use `arcis = "0.8.0"` and `arcium-anchor = "0.8.0"` / `arcium-client = "0.8.0"`.

---

## Table of Contents

1. [Blackjack (Most Complex)](#1-blackjack)
2. [Sealed Bid Auction](#2-sealed-bid-auction)
3. [Voting](#3-voting)
4. [Coinflip](#4-coinflip)
5. [Rock Paper Scissors (Against Player)](#5-rps-against-player)
6. [Rock Paper Scissors (Against House)](#6-rps-against-house)
7. [Ed25519 Signature](#7-ed25519)
8. [Share Medical Records](#8-share-medical-records)
9. [Key Patterns & Observations](#9-key-patterns)

---

## 1. Blackjack

### encrypted-ixs/Cargo.toml
```toml
[package]
name = "encrypted-ixs"
version = "0.1.0"
edition = "2021"

[dependencies]
arcis = "0.8.0"
blake3 = "=1.8.2"
```

### encrypted-ixs/src/lib.rs (Arcis Circuit)
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    const INITIAL_DECK: [u8; 52] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51,
    ];

    const POWS_OF_SIXTY_FOUR: [u128; 21] = [
        1, 64, 4096, 262144, 16777216, 1073741824, 68719476736,
        4398046511104, 281474976710656, 18014398509481984,
        1152921504606846976, 73786976294838206464, 4722366482869645213696,
        302231454903657293676544, 19342813113834066795298816,
        1237940039285380274899124224, 79228162514264337593543950336,
        5070602400912917605986812821504, 324518553658426726783156020576256,
        20769187434139310514121985316880384,
        1329227995784915872903807060280344576,
    ];

    // Pack 52 cards into three u128s (21 + 21 + 10 cards, 6 bits each)
    pub struct Deck {
        pub card_one: u128,
        pub card_two: u128,
        pub card_three: u128,
    }

    impl Deck {
        pub fn from_array(array: [u8; 52]) -> Deck {
            let mut card_one = 0;
            for i in 0..21 {
                card_one += POWS_OF_SIXTY_FOUR[i] * array[i] as u128;
            }
            let mut card_two = 0;
            for i in 21..42 {
                card_two += POWS_OF_SIXTY_FOUR[i - 21] * array[i] as u128;
            }
            let mut card_three = 0;
            for i in 42..52 {
                card_three += POWS_OF_SIXTY_FOUR[i - 42] * array[i] as u128;
            }
            Deck { card_one, card_two, card_three }
        }

        fn to_array(&self) -> [u8; 52] {
            let mut card_one = self.card_one;
            let mut card_two = self.card_two;
            let mut card_three = self.card_three;
            let mut bytes = [0u8; 52];
            for i in 0..21 {
                bytes[i] = (card_one % 64) as u8;
                bytes[i + 21] = (card_two % 64) as u8;
                card_one >>= 6;
                card_two >>= 6;
            }
            for i in 42..52 {
                bytes[i] = (card_three % 64) as u8;
                card_three >>= 6;
            }
            bytes
        }
    }

    pub struct Hand {
        pub cards: u128,
    }

    impl Hand {
        pub fn from_array(array: [u8; 11]) -> Hand {
            let mut cards = 0;
            for i in 0..11 {
                cards += POWS_OF_SIXTY_FOUR[i] * array[i] as u128;
            }
            Hand { cards }
        }

        fn to_array(&self) -> [u8; 11] {
            let mut cards = self.cards;
            let mut bytes = [0u8; 11];
            for i in 0..11 {
                bytes[i] = (cards % 64) as u8;
                cards >>= 6;
            }
            bytes
        }
    }

    #[instruction]
    pub fn shuffle_and_deal_cards(
        mxe: Mxe,
        mxe_again: Mxe,
        client: Shared,
        client_again: Shared,
    ) -> (
        Enc<Mxe, Deck>,
        Enc<Mxe, Hand>,
        Enc<Shared, Hand>,
        Enc<Shared, u8>,
    ) {
        let mut initial_deck = INITIAL_DECK;
        ArcisRNG::shuffle(&mut initial_deck);

        let deck = mxe.from_arcis(Deck::from_array(initial_deck));

        let mut dealer_cards = [53; 11];
        dealer_cards[0] = initial_deck[1];
        dealer_cards[1] = initial_deck[3];
        let dealer_hand = mxe_again.from_arcis(Hand::from_array(dealer_cards));

        let mut player_cards = [53; 11];
        player_cards[0] = initial_deck[0];
        player_cards[1] = initial_deck[2];
        let player_hand = client.from_arcis(Hand::from_array(player_cards));

        (deck, dealer_hand, player_hand, client_again.from_arcis(initial_deck[1]))
    }

    #[instruction]
    pub fn player_hit(
        deck_ctxt: Enc<Mxe, Deck>,
        player_hand_ctxt: Enc<Shared, Hand>,
        player_hand_size: u8,
        dealer_hand_size: u8,
    ) -> (Enc<Shared, Hand>, bool) {
        let deck = deck_ctxt.to_arcis().to_array();
        let mut player_hand = player_hand_ctxt.to_arcis().to_array();
        let player_hand_value = calculate_hand_value(&player_hand, player_hand_size);
        let is_bust = player_hand_value > 21;
        let new_card = if !is_bust {
            let card_index = (player_hand_size + dealer_hand_size) as usize;
            deck[card_index]
        } else {
            53
        };
        player_hand[player_hand_size as usize] = new_card;
        (
            player_hand_ctxt.owner.from_arcis(Hand::from_array(player_hand)),
            is_bust.reveal(),
        )
    }

    #[instruction]
    pub fn player_stand(player_hand_ctxt: Enc<Shared, Hand>, player_hand_size: u8) -> bool {
        let player_hand = player_hand_ctxt.to_arcis().to_array();
        let value = calculate_hand_value(&player_hand, player_hand_size);
        (value > 21).reveal()
    }

    #[instruction]
    pub fn player_double_down(
        deck_ctxt: Enc<Mxe, Deck>,
        player_hand_ctxt: Enc<Shared, Hand>,
        player_hand_size: u8,
        dealer_hand_size: u8,
    ) -> (Enc<Shared, Hand>, bool) {
        let deck = deck_ctxt.to_arcis();
        let deck_array = deck.to_array();
        let mut player_hand = player_hand_ctxt.to_arcis().to_array();
        let player_hand_value = calculate_hand_value(&player_hand, player_hand_size);
        let is_bust = player_hand_value > 21;
        let new_card = if !is_bust {
            let card_index = (player_hand_size + dealer_hand_size) as usize;
            deck_array[card_index]
        } else {
            53
        };
        player_hand[player_hand_size as usize] = new_card;
        (
            player_hand_ctxt.owner.from_arcis(Hand::from_array(player_hand)),
            is_bust.reveal(),
        )
    }

    #[instruction]
    pub fn dealer_play(
        deck_ctxt: Enc<Mxe, Deck>,
        dealer_hand_ctxt: Enc<Mxe, Hand>,
        client: Shared,
        player_hand_size: u8,
        dealer_hand_size: u8,
    ) -> (Enc<Mxe, Hand>, Enc<Shared, Hand>, u8) {
        let deck = deck_ctxt.to_arcis();
        let deck_array = deck.to_array();
        let mut dealer = dealer_hand_ctxt.to_arcis().to_array();
        let mut size = dealer_hand_size as usize;

        for _ in 0..7 {
            let val = calculate_hand_value(&dealer, size as u8);
            if val < 17 {
                let idx = (player_hand_size as usize + size) as usize;
                dealer[size] = deck_array[idx];
                size += 1;
            }
        }

        (
            dealer_hand_ctxt.owner.from_arcis(Hand::from_array(dealer)),
            client.from_arcis(Hand::from_array(dealer)),
            (size as u8).reveal(),
        )
    }

    fn calculate_hand_value(hand: &[u8; 11], hand_length: u8) -> u8 {
        let mut value = 0;
        let mut has_ace = false;
        for i in 0..11 {
            let rank = if i < hand_length as usize { hand[i] % 13 } else { 0 };
            if i < hand_length as usize {
                if rank == 0 {
                    value += 11;
                    has_ace = true;
                } else if rank > 10 {
                    value += 10;
                } else {
                    value += rank;
                }
            }
        }
        if value > 21 && has_ace { value -= 10; }
        value
    }

    #[instruction]
    pub fn resolve_game(
        player_hand: Enc<Shared, Hand>,
        dealer_hand: Enc<Mxe, Hand>,
        player_hand_length: u8,
        dealer_hand_length: u8,
    ) -> u8 {
        let player_hand = player_hand.to_arcis().to_array();
        let dealer_hand = dealer_hand.to_arcis().to_array();
        let player_value = calculate_hand_value(&player_hand, player_hand_length);
        let dealer_value = calculate_hand_value(&dealer_hand, dealer_hand_length);

        let result = if player_value > 21 { 0 }
        else if dealer_value > 21 { 1 }
        else if player_value > dealer_value { 2 }
        else if dealer_value > player_value { 3 }
        else { 4 };

        result.reveal()
    }
}
```

### programs/blackjack/Cargo.toml
```toml
[package]
name = "blackjack"
version = "0.1.0"
description = "Created with Arcium & Anchor"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]
name = "blackjack"

[features]
default = []
cpi = ["no-entrypoint"]
no-entrypoint = []
no-idl = []
no-log-ix-name = []
idl-build = ["anchor-lang/idl-build", "arcium-anchor/idl-build"]

[dependencies]
anchor-lang = { version = "0.32.1", features = ["init-if-needed"] }
arcium-client = { version = "0.8.0", default-features = false }
arcium-macros = "0.8.0"
arcium-anchor = "0.8.0"
```

### Key Anchor Program Patterns (blackjack programs/blackjack/src/lib.rs)

```rust
use anchor_lang::prelude::*;
use arcium_anchor::prelude::*;
use arcium_client::idl::arcium::types::CallbackAccount;

const COMP_DEF_OFFSET_SHUFFLE_AND_DEAL_CARDS: u32 = comp_def_offset("shuffle_and_deal_cards");
// ... one per circuit function

#[arcium_program]
pub mod blackjack {
    // For each circuit function:
    // 1. init_*_comp_def() - initializes computation definition
    // 2. action function - builds args and calls queue_computation
    // 3. *_callback - receives and processes MPC results

    // Example: queue_computation call
    pub fn initialize_blackjack_game(ctx: ..., computation_offset: u64, ...) -> Result<()> {
        let args = ArgBuilder::new()
            .plaintext_u128(mxe_nonce)           // For Mxe params
            .x25519_pubkey(client_pubkey)          // For Shared params
            .plaintext_u128(client_nonce)          // Nonce for Shared
            .build();

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            vec![CallbackIx::callback_ix(computation_offset, &mxe_account, &[CallbackAccount { ... }])],
            1, // num_computations
            0, // computation_index
        )?;
        Ok(())
    }

    // Callback pattern
    #[arcium_callback(encrypted_ix = "shuffle_and_deal_cards")]
    pub fn shuffle_and_deal_cards_callback(
        ctx: Context<Callback>,
        output: SignedComputationOutputs<ShuffleAndDealCardsOutput>,
    ) -> Result<()> {
        let o = output.verify_output(&ctx.accounts.cluster_account, &ctx.accounts.computation_account)?;
        // Access output fields: o.field_0.ciphertexts, o.field_0.nonce, etc.
        Ok(())
    }
}

// Account structures use derive macros:
#[queue_computation_accounts("shuffle_and_deal_cards", payer)]
#[derive(Accounts)]
pub struct InitializeBlackjackGame<'info> { ... }

#[callback_accounts("shuffle_and_deal_cards")]
#[derive(Accounts)]
pub struct ShuffleAndDealCardsCallback<'info> { ... }

#[init_computation_definition_accounts("shuffle_and_deal_cards", payer)]
#[derive(Accounts)]
pub struct InitShuffleAndDealCardsCompDef<'info> { ... }
```

---

## 2. Sealed Bid Auction

### encrypted-ixs/src/lib.rs (Arcis Circuit)
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct Bid {
        pub bidder_lo: u128,     // Pubkey split into two u128s
        pub bidder_hi: u128,
        pub amount: u64,
    }

    pub struct AuctionState {
        pub highest_bid: u64,
        pub highest_bidder_lo: u128,
        pub highest_bidder_hi: u128,
        pub second_highest_bid: u64,
        pub bid_count: u8,
    }

    pub struct AuctionResult {
        pub winner_lo: u128,
        pub winner_hi: u128,
        pub payment_amount: u64,
    }

    #[instruction]
    pub fn init_auction_state(mxe: Mxe) -> Enc<Mxe, AuctionState> {
        let initial_state = AuctionState {
            highest_bid: 0,
            highest_bidder_lo: 0,
            highest_bidder_hi: 0,
            second_highest_bid: 0,
            bid_count: 0,
        };
        mxe.from_arcis(initial_state)
    }

    #[instruction]
    pub fn place_bid(
        bid_ctxt: Enc<Shared, Bid>,
        state_ctxt: Enc<Mxe, AuctionState>,
    ) -> Enc<Mxe, AuctionState> {
        let bid = bid_ctxt.to_arcis();
        let mut state = state_ctxt.to_arcis();

        if bid.amount > state.highest_bid {
            state.second_highest_bid = state.highest_bid;
            state.highest_bid = bid.amount;
            state.highest_bidder_lo = bid.bidder_lo;
            state.highest_bidder_hi = bid.bidder_hi;
        } else if bid.amount > state.second_highest_bid {
            state.second_highest_bid = bid.amount;
        }
        state.bid_count += 1;

        state_ctxt.owner.from_arcis(state)
    }

    #[instruction]
    pub fn determine_winner_first_price(state_ctxt: Enc<Mxe, AuctionState>) -> AuctionResult {
        let state = state_ctxt.to_arcis();
        AuctionResult {
            winner_lo: state.highest_bidder_lo,
            winner_hi: state.highest_bidder_hi,
            payment_amount: state.highest_bid,
        }.reveal()
    }

    #[instruction]
    pub fn determine_winner_vickrey(state_ctxt: Enc<Mxe, AuctionState>) -> AuctionResult {
        let state = state_ctxt.to_arcis();
        AuctionResult {
            winner_lo: state.highest_bidder_lo,
            winner_hi: state.highest_bidder_hi,
            payment_amount: state.second_highest_bid,
        }.reveal()
    }
}
```

### Key Anchor Pattern - Passing encrypted data from client
```rust
// In anchor program - receiving client-encrypted data:
pub fn place_bid(
    ctx: Context<PlaceBid>,
    computation_offset: u64,
    encrypted_bidder_lo: [u8; 32],    // Client-encrypted
    encrypted_bidder_hi: [u8; 32],    // Client-encrypted
    encrypted_amount: [u8; 32],       // Client-encrypted
    bidder_pubkey: [u8; 32],          // x25519 pubkey
    nonce: u128,
) -> Result<()> {
    let args = ArgBuilder::new()
        .x25519_pubkey(bidder_pubkey)
        .plaintext_u128(nonce)
        .encrypted_u128(encrypted_bidder_lo)   // Pre-encrypted by client
        .encrypted_u128(encrypted_bidder_hi)
        .encrypted_u64(encrypted_amount)
        // MXE-encrypted state from on-chain account
        .plaintext_u128(auction.state_nonce)
        .account(ctx.accounts.auction.key(), OFFSET, SIZE)
        .build();
    // ...
}
```

### TypeScript Client - Client-side encryption for sealed bid
```typescript
import { RescueCipher, x25519, deserializeLE } from "@arcium-hq/client";

// Setup encryption
const privateKey = x25519.utils.randomSecretKey();
const publicKey = x25519.getPublicKey(privateKey);
const mxePublicKey = await getMXEPublicKey(provider, programId);
const sharedSecret = x25519.getSharedSecret(privateKey, mxePublicKey);
const cipher = new RescueCipher(sharedSecret);

// Split pubkey into two u128s
function splitPubkeyToU128s(pubkey: Uint8Array): { lo: bigint; hi: bigint } {
  const lo = deserializeLE(pubkey.slice(0, 16));
  const hi = deserializeLE(pubkey.slice(16, 32));
  return { lo, hi };
}

// Encrypt bid data
const bidAmount = BigInt(500);
const nonce = randomBytes(16);
const bidPlaintext = [bidderLo, bidderHi, bidAmount];
const bidCiphertext = cipher.encrypt(bidPlaintext, nonce);

// Send to program
await program.methods.placeBid(
    computationOffset,
    Array.from(bidCiphertext[0]),     // encrypted_bidder_lo
    Array.from(bidCiphertext[1]),     // encrypted_bidder_hi
    Array.from(bidCiphertext[2]),     // encrypted_amount
    Array.from(publicKey),
    new anchor.BN(deserializeLE(nonce).toString())
).rpc();
```

---

## 3. Voting

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct VoteStats { yes: u64, no: u64 }
    pub struct UserVote { vote: bool }

    #[instruction]
    pub fn init_vote_stats(mxe: Mxe) -> Enc<Mxe, VoteStats> {
        let vote_stats = VoteStats { yes: 0, no: 0 };
        mxe.from_arcis(vote_stats)
    }

    #[instruction]
    pub fn vote(
        vote_ctxt: Enc<Shared, UserVote>,
        vote_stats_ctxt: Enc<Mxe, VoteStats>,
    ) -> Enc<Mxe, VoteStats> {
        let user_vote = vote_ctxt.to_arcis();
        let mut vote_stats = vote_stats_ctxt.to_arcis();
        if user_vote.vote { vote_stats.yes += 1; }
        else { vote_stats.no += 1; }
        vote_stats_ctxt.owner.from_arcis(vote_stats)
    }

    #[instruction]
    pub fn reveal_result(vote_stats_ctxt: Enc<Mxe, VoteStats>) -> bool {
        let vote_stats = vote_stats_ctxt.to_arcis();
        (vote_stats.yes > vote_stats.no).reveal()
    }
}
```

---

## 4. Coinflip

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct UserChoice { pub choice: bool }

    #[instruction]
    pub fn flip(input_ctxt: Enc<Shared, UserChoice>) -> bool {
        let input = input_ctxt.to_arcis();
        let toss = ArcisRNG::bool();
        (input.choice == toss).reveal()
    }
}
```

### TypeScript Client (Complete)
```typescript
import { RescueCipher, x25519, deserializeLE, getMXEPublicKey } from "@arcium-hq/client";

// Encryption setup
const privateKey = x25519.utils.randomSecretKey();
const publicKey = x25519.getPublicKey(privateKey);
const mxePublicKey = await getMXEPublicKey(provider, programId);
const sharedSecret = x25519.getSharedSecret(privateKey, mxePublicKey);
const cipher = new RescueCipher(sharedSecret);

// Encrypt choice
const choice = BigInt(true);
const nonce = randomBytes(16);
const ciphertext = cipher.encrypt([choice], nonce);

// Queue computation
await program.methods.flip(
    computationOffset,
    Array.from(ciphertext[0]),    // encrypted choice
    Array.from(publicKey),         // x25519 pubkey
    new anchor.BN(deserializeLE(nonce).toString())
).accountsPartial({ ... }).rpc();

// Wait for result
const finalizeSig = await awaitComputationFinalization(provider, computationOffset, programId, "confirmed");
const flipEvent = await flipEventPromise;
console.log(flipEvent.result ? "Won!" : "Lost!");
```

---

## 5. RPS Against Player

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct GameMoves { player_a_move: u8, player_b_move: u8 }
    pub struct PlayersMove { player: u8, player_move: u8 }

    #[instruction]
    pub fn init_game(mxe: Mxe) -> Enc<Mxe, GameMoves> {
        let game_moves = GameMoves { player_a_move: 3, player_b_move: 3 };
        mxe.from_arcis(game_moves)
    }

    #[instruction]
    pub fn player_move(
        players_move_ctxt: Enc<Shared, PlayersMove>,
        game_ctxt: Enc<Mxe, GameMoves>,
    ) -> Enc<Mxe, GameMoves> {
        let players_move = players_move_ctxt.to_arcis();
        let mut game_moves = game_ctxt.to_arcis();
        if players_move.player == 0 && game_moves.player_a_move == 3 && players_move.player_move < 3 {
            game_moves.player_a_move = players_move.player_move;
        } else if players_move.player == 1 && game_moves.player_b_move == 3 && players_move.player_move < 3 {
            game_moves.player_b_move = players_move.player_move;
        }
        game_ctxt.owner.from_arcis(game_moves)
    }

    #[instruction]
    pub fn compare_moves(game_ctxt: Enc<Mxe, GameMoves>) -> u8 {
        let game_moves = game_ctxt.to_arcis();
        let result = if game_moves.player_a_move == 3 || game_moves.player_b_move == 3 { 3 }
        else if game_moves.player_a_move == game_moves.player_b_move { 0 }
        else if (game_moves.player_a_move == 0 && game_moves.player_b_move == 2) ||
                (game_moves.player_a_move == 1 && game_moves.player_b_move == 0) ||
                (game_moves.player_a_move == 2 && game_moves.player_b_move == 1) { 1 }
        else { 2 };
        result.reveal()
    }
}
```

---

## 6. RPS Against House

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct PlayerMove { player_move: u8 }

    #[instruction]
    pub fn play_rps(player_move_ctxt: Enc<Shared, PlayerMove>) -> u8 {
        let player_move = player_move_ctxt.to_arcis();
        let first_bit = ArcisRNG::bool();
        let second_bit = ArcisRNG::bool();
        let house_move = if first_bit {
            if second_bit { 0 } else { 2 }
        } else if second_bit { 1 } else { 0 };

        let result = if player_move.player_move > 2 { 3 }
        else if player_move.player_move == house_move { 0 }
        else if (player_move.player_move == 0 && house_move == 2) ||
                (player_move.player_move == 1 && house_move == 0) ||
                (player_move.player_move == 2 && house_move == 1) { 1 }
        else { 2 };
        result.reveal()
    }
}
```

---

## 7. Ed25519

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    #[instruction]
    pub fn sign_message(message: [u8; 5]) -> ArcisEd25519Signature {
        let signature = MXESigningKey::sign(&message);
        signature.reveal()
    }

    #[instruction]
    pub fn verify_signature(
        verifying_key_enc: Enc<Shared, Pack<VerifyingKey>>,
        message: [u8; 5],
        signature: [u8; 64],
        observer: Shared,
    ) -> Enc<Shared, bool> {
        let verifying_key = verifying_key_enc.to_arcis().unpack();
        let signature = ArcisEd25519Signature::from_bytes(signature);
        let is_valid = verifying_key.verify(&message, &signature);
        observer.from_arcis(is_valid)
    }
}
```

---

## 8. Share Medical Records

### encrypted-ixs/src/lib.rs
```rust
use arcis::*;

#[encrypted]
mod circuits {
    use arcis::*;

    pub struct PatientData {
        pub patient_id: u64,
        pub age: u8,
        pub gender: bool,
        pub blood_type: u8,
        pub weight: u16,
        pub height: u16,
        pub allergies: [bool; 5],
    }

    #[instruction]
    pub fn share_patient_data(
        receiver: Shared,
        input_ctxt: Enc<Shared, PatientData>,
    ) -> Enc<Shared, PatientData> {
        let input = input_ctxt.to_arcis();
        receiver.from_arcis(input)
    }
}
```

---

## 9. Key Patterns & Observations

### A. Encryption Ownership Model

Two key types control who can decrypt data:

| Type | Meaning | When to use |
|------|---------|------------|
| `Mxe` | MXE network holds the key | Persistent server-side state (deck, game state, vote tallies) |
| `Shared` | Client holds the decryption key | Data the client needs to read (player's hand, their vote) |

### B. Input/Output Patterns

**Inputs to `#[instruction]` functions:**

| Pattern | Meaning |
|---------|---------|
| `Enc<Shared, T>` | Client-encrypted data (bid, vote, move) |
| `Enc<Mxe, T>` | MXE-encrypted state from on-chain account |
| `Mxe` | MXE key handle - use to encrypt outputs for MXE storage |
| `Shared` | Client key handle - use to encrypt outputs for a client |
| `u8`, `u64`, etc. | Plaintext parameters (hand size, etc.) |

**Outputs from `#[instruction]` functions:**

| Pattern | Meaning |
|---------|---------|
| `Enc<Mxe, T>` | Encrypted for MXE - stored on-chain |
| `Enc<Shared, T>` | Encrypted for client - client can decrypt |
| `T` (with `.reveal()`) | Plaintext output - visible to everyone |
| Tuples | Multiple outputs of mixed types |

### C. How .reveal() Works

```rust
// Reveal a primitive - makes it plaintext in the output
result.reveal()      // u8, bool, etc.

// Reveal a struct - all fields become plaintext
AuctionResult { ... }.reveal()

// Reveal a comparison
(value > 21).reveal()
```

### D. How Arrays/Matrices Are Handled

Arrays of encrypted values are NOT directly supported as struct fields in the encrypted type system. Instead, arrays are **packed into u128 values** using base-64 encoding:

```rust
// Pack 52 cards (6 bits each) into three u128s
pub struct Deck {
    pub card_one: u128,   // cards 0-20 (21 cards * 6 bits = 126 bits)
    pub card_two: u128,   // cards 21-41
    pub card_three: u128, // cards 42-51
}

// Pack 11 cards into one u128
pub struct Hand {
    pub cards: u128,      // 11 cards * 6 bits = 66 bits
}
```

Boolean arrays ARE supported:
```rust
pub struct PatientData {
    pub allergies: [bool; 5],   // This works
}
```

### E. How Multiplication Works

Multiplication on encrypted types works naturally using standard Rust operators:

```rust
// u128 multiplication in encrypted context
card_one += POWS_OF_SIXTY_FOUR[i] * array[i] as u128;

// These all work inside #[encrypted] blocks:
value += rank;         // addition
state.bid_count += 1;  // increment
card_one >>= 6;        // shift
card_one % 64          // modulo
```

### F. Re-encrypting Data for Different Recipients

```rust
// Transfer data from one party to another:
// 1. Decrypt from sender's encryption
let input = input_ctxt.to_arcis();
// 2. Re-encrypt for receiver
receiver.from_arcis(input)

// Use .owner to re-encrypt for the same party:
player_hand_ctxt.owner.from_arcis(Hand::from_array(updated_hand))
// This preserves the original encryption key
```

### G. Random Number Generation

```rust
ArcisRNG::bool()              // Random boolean
ArcisRNG::shuffle(&mut arr)   // Fisher-Yates shuffle of array
```

### H. ArgBuilder Pattern (Anchor Program Side)

```rust
let args = ArgBuilder::new()
    // For Mxe params:
    .plaintext_u128(mxe_nonce)

    // For Shared params:
    .x25519_pubkey(client_pubkey)
    .plaintext_u128(client_nonce)

    // Pre-encrypted data from client:
    .encrypted_u128(encrypted_value)
    .encrypted_u64(encrypted_amount)
    .encrypted_u8(encrypted_choice)
    .encrypted_bool(encrypted_vote)

    // On-chain encrypted state (read from account):
    .plaintext_u128(state_nonce)
    .account(account_key, byte_offset, byte_size)

    // Plaintext parameters:
    .plaintext_u8(hand_size)

    .build();
```

### I. Callback Output Destructuring

```rust
// Single encrypted output
Ok(InitAuctionStateOutput { field_0 }) => {
    // field_0.ciphertexts: [[u8; 32]; N]  (N = number of struct fields)
    // field_0.nonce: u128
}

// Tuple of mixed outputs
Ok(PlayerHitOutput { field_0: PlayerHitOutputStruct0 { field_0: hand, field_1: is_bust } }) => {
    // hand.ciphertexts, hand.nonce  (encrypted Shared output)
    // is_bust: bool                 (revealed plaintext)
}

// Struct fields for Enc<Shared, T> output:
// .encryption_key: [u8; 32]   (client's x25519 pubkey)
// .nonce: u128
// .ciphertexts: [[u8; 32]; N]
```

### J. TypeScript Decryption Pattern

```typescript
const cipher = new RescueCipher(sharedSecret);
const nonce = Uint8Array.from(event.clientNonce.toArray("le", 16));
const decrypted = cipher.decrypt([event.playerHand], nonce);
// decrypted[0] is a BigInt containing the packed value
```

### K. Project Structure

```
example/
  Arcium.toml                    # Localnet config (nodes, backends)
  Cargo.toml                     # Workspace
  encrypted-ixs/
    Cargo.toml                   # arcis dependency only
    src/lib.rs                   # #[encrypted] circuit definitions
  programs/example/
    Cargo.toml                   # anchor-lang + arcium-* dependencies
    src/lib.rs                   # #[arcium_program] Anchor program
  tests/
    example.ts                   # TypeScript integration tests
```

### L. On-Chain State for Encrypted Data

Each encrypted field maps to `[u8; 32]` on-chain. A struct with N fields
becomes `[[u8; 32]; N]` on-chain:

```rust
// Arcis circuit struct (5 fields):
pub struct AuctionState { highest_bid: u64, highest_bidder_lo: u128, ... }

// On-chain Anchor account:
pub encrypted_state: [[u8; 32]; 5],  // 5 ciphertexts, one per field
pub state_nonce: u128,               // nonce for decryption
```
