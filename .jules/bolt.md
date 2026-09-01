## Performance Optimization Insights

### Issue
The `session.idle` hook in `.opencode/plugins/ecc-hooks.ts` was executing a shell command (`grep -c "console\.log"`) sequentially using `await` inside a `for...of` loop over the set of edited files. This resulted in O(N) linear time scaling based on the number of edited files since each command was blocked until the previous one completed.

### Resolution
Changed the sequential looping to a parallel execution strategy using `Promise.all`:
1. Mapped the `Set` of edited files into an array of concurrent `Promise`s.
2. Filtered out non-relevant files first.
3. Used `Promise.all` to await all shell executions simultaneously.
4. Accumulated the results iterating over the resolved promises to maintain the original data formatting structure (`totalConsoleLogCount`, `filesWithConsoleLogs`).

### Benchmark
A benchmark was created using 50 placeholder TypeScript files:
- **Sequential Baseline:** ~359 ms
- **Parallel Optimization:** ~77 ms
- **Improvement:** ~4.7x speedup (decreased latency by ~79%)
