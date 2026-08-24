# Hybrid Deep Learning GARCH for Stablecoin Volatility Forecasting - Model Rerun

This is the log for rerunning this project. I will keep updates in here, before committing to organize everything (potentially) under a new folder, new notebooks, second paper etc. This file is ordered in reverse chronological order, such that new updates are at the top.

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