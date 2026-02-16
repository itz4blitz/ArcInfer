// Browser stub for Node's fs module.
// @arcium-hq/client imports this at top level but never uses it in browser.
const noop = () => {
  throw new Error("Node fs is not available in browser");
};
export const readFileSync = noop;
export const writeFileSync = noop;
export const existsSync = () => false;
export default { readFileSync, writeFileSync, existsSync };
