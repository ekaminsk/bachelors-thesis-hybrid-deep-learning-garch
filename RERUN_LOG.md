# Hybrid Deep Learning GARCH for Stablecoin Volatility Forecasting - Model Rerun

This is the log for rerunning this project. I will keep updates in here before committing to organizing everything (potentially) under a new folder, new notebooks, a second paper, etc. This file is ordered in reverse chronological order, such that new updates are at the top.

## 25.08.2026 - Designing historical data collection system

Immediately I think that I need a Dune query to tell me the exact block number at the end of a 5 min window, which feeds it into the data collection code. 




## 24.08.2026 - Figuring out if I can somehow pull data historically

So it appears that I might be able to filter for blocks in the graphql query. The pool query would look the following
```
query PoolAtBlock($id: ID!, $block: Int!) {
  pool(id: $id, block: { number: $block }) {
    liquidity
    tick
    sqrtPrice
    token0Price
    token1Price
    totalValueLockedToken0
    totalValueLockedToken1
    totalValueLockedUSD
  }
}
```
The only question is how far the indexing goes. It is possible that for storage cost reasons, blocks are cut after a certain time. This is the first test.

Using the following query in thegraph.com's explorer feature, I can find that the earliest, still indexed block is **12369621**. This is still the USDT/USDC pool. I find the block number by inputing number: 0 and then receiving 'startBlock must be 12369621' (paraphrased)

Using a simple query on Dune Analytics, I find that this block is from 2021-05-04 19:27:00.
```
SELECT
    time,
    number
FROM ethereum.blocks
WHERE number = 12369621
```

Further testing reveals that ticks, pool data, and mints / burns are indeed historically queryable by adding the block identifier. Especially from 2023 onwards (from block 16308190). The only thing that is not queryable and will never be (without third-party services) is CEX orderbook data. But I am willing to sacrifice that signal to test the rest on a better, longer window.