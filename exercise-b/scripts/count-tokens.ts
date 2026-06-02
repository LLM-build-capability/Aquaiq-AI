  import { get_encoding } from "tiktoken";
  import { TOOLS, DSL_BOOTSTRAP } from "../src/index.js";

  const BASELINE_TOKENS = 1417;
  const TARGET_TOKENS = 1200;

  function countTokens(text: string): number {
    const enc = get_encoding("cl100k_base");
    const n = enc.encode(text).length;
    enc.free();
    return n;
  }

  function main() {
    const toolsJson = JSON.stringify(TOOLS, null, 2);
    const toolsTokens = countTokens(toolsJson);
    const dslTokens = countTokens(DSL_BOOTSTRAP);
    const combined = toolsTokens + dslTokens;

    console.log("Token Measurement (cl100k_base)");
    console.log("--------------------------------");
    console.log(`Tools JSON:      ${toolsTokens} tokens`);
    console.log(`DSL bootstrap:   ${dslTokens} tokens`);
    console.log(`Combined:        ${combined} tokens`);
    console.log("");
    console.log("Comparison");
    console.log("----------");
    console.log(`N-tool baseline: ${BASELINE_TOKENS} tokens`);
    console.log(`Code Mode:       ${combined} tokens`);
    console.log(`Delta:           ${combined - BASELINE_TOKENS} (${Math.round((combined / BASELINE_TOKENS - 1) * 100)}%)`);
    console.log("");
    console.log(
      combined <= TARGET_TOKENS
        ? `Status: within target (≤ ${TARGET_TOKENS})`
        : `Status: above target (≤ ${TARGET_TOKENS})`,
    );
  }

  main();

