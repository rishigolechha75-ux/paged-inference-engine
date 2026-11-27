# Walkthrough

This document explains the project from first principles so you can defend every line of it in an interview.

## 1. The problem and why it matters

An autoregressive LLM generates one token at a time. Each new token attends to every earlier token, so the engine keeps the keys and values (K/V) of all earlier tokens in memory: the **KV cache**. For GPT-2 small in float32 that is `2 (K and V) x 12 layers x 768 dims x 4 bytes = 73,728 bytes per token`. A 1,000-token conversation needs 72 MB of cache, on top of the model weights.

Two things decide how fast a serving system is:

1. **Batch size.** A decode step for one sequence is a matrix-vector product: it reads all 124M weights to do very little arithmetic. Running 32 sequences in the same step reads the weights once and does 32x the work, so throughput keeps rising with batch size until the hardware is compute bound. On this laptop CPU, the same engine goes from 127.9 tok/s at batch 1 to 1,083.9 tok/s at batch 64 (`reports/batch_scaling.json`). Batch 2 came out slightly slower than batch 1 (113.9 tok/s), which is measurement noise at sub-second run lengths, not a real effect.
2. **How many sequences fit in KV memory at once.** The batch can only be as large as the number of sequences whose caches fit.

Naive servers lose on both:

- **Static batching** waits for a whole batch to finish before it starts the next. Output lengths vary a lot (Dolly responses range from 4 to 256+ tokens here), so most of a batch's lifetime is spent running a few stragglers at a tiny batch size while new requests queue.
- **Contiguous, pre-reserved KV memory** has to reserve `prompt + max_tokens` slots per request because the true output length is unknown up front. Most of that reservation is never used, which caps concurrency.

This project implements the two fixes that modern servers (vLLM, TGI, SGLang, TensorRT-LLM) are built on, measures each one separately, and checks that the output is still exactly what the original model produces.

- **Continuous (iteration-level) batching**, from the Orca paper: re-decide the batch every step, so a finished sequence's slot is refilled immediately.
- **PagedAttention**, from the vLLM paper: store the KV cache in fixed-size blocks allocated on demand, the way an OS pages virtual memory, so waste is limited to the unfilled part of each sequence's last block.

Plus two refinements: **chunked prefill** (a long prompt is processed in pieces so it does not stall ongoing decodes) and **prefix caching** (requests that share a prompt prefix share its KV blocks).

## 2. Architecture and data flow

```
HTTP client ──POST /v1/completions──> FastAPI handler
                                        │ tokenizer.encode, add_request (under lock)
                                        ▼
                               ┌──────── LLMEngine (background thread) ────────┐
                               │ loop:                                          │
                               │  1. Scheduler.schedule()                       │
                               │     - running seqs first (decode=1 token,      │
                               │       prefill chunk = remaining budget)        │
                               │     - preempt newest if no free block          │
                               │     - admit waiting FIFO with leftover budget  │
                               │  2. _build_batch(): flat tokens, slot mapping, │
                               │     attention groups, block tables             │
                               │  3. GPT2.forward_paged(): per layer            │
                               │     scatter new K/V -> gather via block table  │
                               │     -> masked SDPA                             │
                               │  4. sample last-token logits, append, finish?  │
                               └────────────────────────────────────────────────┘
                                        │ asyncio.Queue per request (call_soon_threadsafe)
                                        ▼
                               SSE token stream / JSON response
```

The memory layout:

```
KV pool, one per layer:  k[num_blocks, block_size, heads, head_dim]  (same for v)

seq A block_table = [7, 2, 9]   tokens 0-15 -> block 7, 16-31 -> block 2, 32-40 -> block 9 (partly filled)
seq B block_table = [7, 4]      shares block 7 with A (same first 16 tokens, ref_count = 2)
```

Token `i` of a sequence lives at flat slot `block_table[i // block_size] * block_size + i % block_size`.

## 3. File-by-file tour

| File | What it does |
|---|---|
| `pagedserve/config.py` | `ModelConfig` (GPT-2 shape, plus a tiny config for tests) and `EngineConfig` (block size, block count, token budget, policy, reserve mode, prefix caching). |
| `pagedserve/sequence.py` | `SamplingParams` (validated) and `Sequence`: prompt, outputs, block table, `num_computed_tokens`, timestamps. `is_prefill` and `num_uncomputed_tokens` drive scheduling. |
| `pagedserve/block_manager.py` | `BlockAllocator`: free list, reference counts, hash table of cached full blocks, LRU of evictable cached blocks. `BlockManager`: per-sequence admission, growth, commit (register full blocks for reuse) and free. `block_hashes` computes chained block hashes. |
| `pagedserve/scheduler.py` | Iteration-level scheduler with a per-step token budget. Running first, recompute-preemption of the newest sequence when blocks run out, then FIFO admission. `static` policy only admits when the running set is empty. |
| `pagedserve/model.py` | GPT-2 in plain PyTorch. `forward_paged` runs the linear layers on a flat `[T, E]` tensor and attention per group with a positional causal mask. `forward_dense` is the cache-free reference. `load_gpt2` maps Hugging Face Conv1D weights (stored `[in, out]`) into `nn.Linear` (`[out, in]`). |
| `pagedserve/engine.py` | `LLMEngine.step()`: schedule, build tensors, forward, sample, finish. `generate()` for offline use, stats collection, and `generate_dense` (ground truth). |
| `pagedserve/sampler.py` | Greedy, temperature, top-k, top-p. Seeded requests get their own `torch.Generator`, so a request's sample does not depend on its batch-mates. |
| `pagedserve/server.py` | FastAPI app. One engine thread; handlers communicate through per-request `asyncio.Queue`s. SSE streaming, `/metrics` in Prometheus text format, `/health`. |
| `pagedserve/tokenizer.py` | GPT-2 BPE from `tokenizer.json`, plus a byte tokenizer for the tiny test model. |
| `pagedserve/cli.py` | `pagedserve generate` and `pagedserve serve`. |
| `bench/workload.py` | Downloads Dolly-15k and builds requests with real prompt and response lengths; builds the shared-prefix few-shot workload. |
| `bench/harness.py` | Replays a workload with Poisson arrivals in wall-clock time and computes TTFT, TPOT, end-to-end latency, throughput, batch size and KV utilisation. |
| `bench/run_benchmarks.py` | Runs every experiment and writes `reports/*.json` and charts. |
| `bench/parity.py` | Greedy outputs vs Hugging Face `transformers` on 24 Dolly prompts. |
| `tests/` | Block manager unit tests, engine-vs-dense-reference tests under seven configurations, scheduler-policy behaviour tests, API tests, and opt-in real GPT-2 parity tests. |

## 4. Key design decisions and trade-offs

**Correctness first: everything is checked against a dense reference.** `generate_dense` re-runs the full sequence through a plain causal transformer at every step. It is slow, but it obviously gives the right answer. The paged engine must reproduce it token for token under block size 1, 3 and 16, chunked prefill with a 7-token budget, static and continuous policies, reserved and lazy allocation, prefix caching, and a memory budget small enough to force preemption. Most bugs in KV-cache code (wrong slot, off-by-one positions, stale blocks, a missing mask) show up as diverging tokens, so this one test family covers a lot.

**One code path for prefill, chunked prefill, recompute and decode.** Each sequence contributes `n` new tokens at positions `[c, c+n)`. The new K/V are written first, then the whole context `[0, c+n)` is gathered and a mask `key_pos <= query_pos` is applied. A decode is just `n = 1`. Alternative: separate prefill and decode kernels, as in production. That is faster on GPU, but it doubles the surface for bugs and is unnecessary here.

**Flat tokens plus attention groups instead of a padded `[B, Q]` batch.** The first version padded every sequence to the longest query in the step. With chunked prefill that is a disaster: one 512-token chunk would pad 31 decodes to 512 tokens each. Now the linear layers (most of the FLOPs) run on exactly `T` real tokens, and only attention is padded, separately for the decode group (Q = 1) and the prefill group. Production engines go further with varlen attention kernels (FlashAttention `cu_seqlens`, the vLLM paged attention kernel) that need no padding at all.

**Gather-then-attend instead of a custom paged attention kernel.** `k_cache[block_tables]` copies each sequence's blocks into a contiguous tensor and then calls PyTorch SDPA. A real PagedAttention kernel reads blocks in place without the copy. The gather costs memory bandwidth, but it is portable (CPU, MPS, CUDA), easy to read, and the scheduling and memory behaviour being measured are the same.

**Lazy allocation reserves the whole prompt at admission.** The paged engine allocates prompt blocks when a request is admitted and then one block at a time while decoding. An alternative is to allocate per prefill chunk, which admits more aggressively but preempts more under pressure. Whole-prompt reservation keeps the logic simple; chunking still bounds compute per step.

**Recompute-style preemption.** When a running sequence needs a block and none is free, the newest running sequence loses its blocks and goes back to the front of the waiting queue; later its prompt and generated tokens are re-prefilled. The alternative is swapping blocks to host memory. On CPU there is no separate host memory to swap to, and for short sequences recomputing is cheap. Preempting the newest sequence (LIFO) protects the oldest requests' latency and avoids starvation.

**Prefix caching with chained hashes.** Block `i`'s hash is `hash(hash_{i-1}, tokens of block i)`, so a block's identity includes its whole prefix and identical text at different positions never aliases. Only full blocks are cached. The last prompt token is never served from cache, because its logits are needed to sample the first output token. Freed cached blocks are kept in an LRU and evicted only when memory is needed. Blocks are freed tail first, so the shared head of a prompt stays cached longest. Alternative: SGLang's RadixAttention keeps a radix tree over token sequences, which also handles partial-block matches and is better for multi-turn branching.

**The baseline uses the same model code.** Static batching and reserved memory are policies of the same engine (`policy="static"`, `reserve_max=True`), not a different library. Any speed difference therefore comes from scheduling and memory management, not from a different matmul implementation.

**Replaying real output lengths.** Every request sends the same client-side cap `max_tokens = 256`, which a reserving allocator has to plan for. The request then stops at its real Dolly response length (`stop_after`), as if the model had emitted EOS. This is how real traffic behaves: the server never knows the true length in advance. Using `ignore_eos` with `max_tokens = true length`, as some benchmarks do, would quietly hand the reserving baseline perfect knowledge.

**Thread plus asyncio queues in the server.** The forward pass is synchronous, CPU-bound PyTorch, so it runs on its own thread and the event loop stays free to accept requests and stream tokens. `loop.call_soon_threadsafe` is the safe way to hand data from a thread to asyncio.

## 5. How the evaluation works and how to read it

All numbers come from `python -m bench.run_benchmarks` and `python -m bench.parity` and are saved in `reports/`. The README table quotes those files directly.

- **Workload:** 160 Dolly-15k requests, real prompt lengths (clipped at 512) and real response lengths (clipped to 4-256 tokens), GPT-2 124M, float32, CPU, KV budget of 384 blocks x 16 = 6,144 token slots (432 MiB).
- **Three systems:** `static_reserve` (classic static batching with a contiguous, reserved cache), `continuous_reserve` (Orca-style scheduling, still reserved), `continuous_paged` (scheduling plus paging). Comparing the first two isolates the scheduling effect; comparing the last two isolates the memory effect.
- **Offline throughput:** every request arrives at t = 0 and output tokens per second are measured over the whole run.
- **Online latency:** Poisson arrivals at 2, 4, 8 and 16 req/s. **TTFT** (time to first token) is mostly queueing plus prefill. **TPOT** (time per output token) is the decode speed a user sees while streaming. **E2E** is the full request latency. Always read p99 next to p50: static batching hurts the tail most.
- **KV utilisation:** tokens actually stored divided by token slots allocated, averaged over steps. Low utilisation means memory is reserved but empty, which directly limits batch size.
- **Fragmentation:** an exact calculation over the workload of allocated vs used slots at completion, for reservation and for each block size.
- **Block-size sweep:** smaller blocks waste less memory; larger blocks mean fewer, bigger gathers.
- **Prefix caching:** 96 requests sharing an 8-shot few-shot prefix, at 8 req/s, with and without the cache. Compare prefill tokens computed and TTFT.
- **What paging costs:** in the offline run the paged engine preempted 57 times and computed 16,342 prefill tokens instead of 12,024 (35.9% extra work from recomputation). Its mean TPOT is also higher (100.65 ms vs 25.28 ms for static batching), because each step runs a much bigger batch. It wins anyway, because the steps carry far more tokens. A good answer in an interview names these costs before being asked.
- **Parity:** greedy outputs from the batched, paged, chunked engine compared with `transformers` generating one prompt at a time.

Caveats to state in an interview: wall-clock numbers on a laptop CPU are noisy (other processes, thermal throttling) and absolute throughput is far below a GPU. The relative comparisons are the result, not the absolute tok/s. On a GPU, batching gains are larger, because a single decode leaves the GPU far more idle than it leaves a CPU.

## 6. Likely interview questions

1. **What is the KV cache and why is it needed?**
   Attention for token t needs the keys and values of all tokens before it. Recomputing them at every step costs O(t) work per layer per step, so a whole generation would be quadratic. Caching them makes each decode step process only the one new token.

2. **Why does batching speed up decoding?**
   A single-token decode is memory-bandwidth bound: every weight is read to do one multiply-add per row. Batching B sequences reuses each weight load B times, until arithmetic becomes the bottleneck.

3. **What problem does continuous batching solve?**
   With static batching, a batch runs until its longest sequence finishes, so slots sit idle and new requests wait. Scheduling per iteration lets a finished sequence's slot go to a waiting request on the very next step.

4. **What is PagedAttention?**
   The KV cache is split into fixed-size blocks, and each sequence has a block table mapping logical token positions to physical blocks. Memory is allocated on demand, so waste is under one block per sequence, and blocks can be shared between sequences through reference counting.

5. **Internal vs external fragmentation here?**
   Internal fragmentation is the unused tail of a sequence's last block (on average about half a block). External fragmentation disappears, because any free block can serve any sequence; no contiguous region is needed. Reserved allocation suffers from over-reservation instead: slots set aside for tokens that are never generated.

6. **How do you choose block size?**
   Smaller blocks waste less memory and give finer-grained prefix sharing, but mean bigger block tables and more scattered memory reads. vLLM defaults to 16. `reports/block_size_sweep.json` and `fragmentation.json` show the trade-off on this workload.

7. **What happens when the KV cache is full?**
   The scheduler preempts the newest running sequence: its blocks are freed and it is re-queued at the front. When it comes back it re-prefills its prompt plus the tokens generated so far. The alternative is swapping blocks to CPU memory.

8. **What is chunked prefill and why does it matter?**
   A long prompt's prefill can take hundreds of milliseconds. If it runs as one step, every decoding user sees a stall. With a token budget per step, the prompt is split into chunks, and decodes, which are scheduled first, keep flowing. It trades a slightly longer TTFT for the long prompt for a smooth TPOT for everyone else.

9. **How does prefix caching work, and why chained hashes?**
   Each full block is keyed by a hash of its tokens combined with the previous block's hash. A new request walks its prompt's block hashes and reuses blocks while they hit. Chaining ensures that two blocks with the same tokens but different preceding context are never confused, because the K/V for a token depend on everything before it.

10. **Why can't the last prompt token come from the cache?**
    The engine needs that token's output logits to sample the first generated token, and the cache stores only K/V, not logits. So at least one token must go through the model.

11. **How do you know the engine is correct?**
    Two levels. In CI, a tiny random-weight GPT-2 must match a cache-free dense implementation token for token under seven configurations, including forced preemption and chunking. Locally, real GPT-2 greedy outputs match Hugging Face `transformers` on 24 Dolly prompts (`reports/parity.json`).

12. **Why is seeded sampling batch-independent here?**
    Each seeded request has its own `torch.Generator`. A single global RNG would make a request's samples depend on which other requests were drawing random numbers in the same step.

13. **Explain TTFT vs TPOT and which one each optimisation affects.**
    TTFT is queueing plus prefill, and continuous batching cuts the queueing part dramatically. TPOT is the per-step decode time, which rises slightly with batch size. Prefix caching reduces TTFT by skipping prefill work. Chunked prefill protects TPOT.

14. **What would you change to run this on a GPU at scale?**
    Use a fused paged attention kernel instead of gather plus SDPA, fp16/bf16 weights and cache, CUDA graphs for decode steps, tensor parallelism for bigger models, swap-to-host preemption, and an async tokenizer/detokenizer process. Speculative decoding and quantised KV are further steps.

15. **What are the limitations of this implementation?**
    It is CPU-oriented, has no custom kernel, handles one model family (GPT-2), has no beam search (so no copy-on-write blocks), uses whole-prompt reservation, and is single-process.

## 7. Concepts to study further

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023): https://arxiv.org/abs/2309.06180
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022): https://www.usenix.org/conference/osdi22/presentation/yu
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (chunked prefill, OSDI 2024): https://arxiv.org/abs/2403.02310
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention prefix caching): https://arxiv.org/abs/2312.07104
- Dao et al., *FlashAttention* (IO-aware attention kernels): https://arxiv.org/abs/2205.14135
- Pope et al., *Efficiently Scaling Transformer Inference* (memory-bound decode analysis): https://arxiv.org/abs/2211.05102
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- vLLM design docs on the scheduler and prefix caching: https://docs.vllm.ai/en/latest/design/
- Operating-system paging and virtual memory, the analogy PagedAttention borrows (any OS textbook, e.g. *Operating Systems: Three Easy Pieces*, chapters 18-23): https://pages.cs.wisc.edu/~remzi/OSTEP/
