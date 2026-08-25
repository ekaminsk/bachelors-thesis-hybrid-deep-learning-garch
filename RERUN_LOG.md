# Hybrid Deep Learning GARCH for Stablecoin Volatility Forecasting - Model Rerun

This is the log for rerunning this project. I will keep updates in here before committing to organizing everything (potentially) under a new folder, new notebooks, a second paper, etc. This file is ordered in reverse chronological order, such that new updates are at the top.

## 25.08.2026 - Designing historical data collection system

Immediately, I thnik that I need a Dune query to output the corresponding block to each 5min interval aligned to UTC clock. This will then be inputted into a new [univ3_pool_historical.py](code/data/univ3_pool_historical.py). The issue now is that I am running into the same Dune credit constraint I had while querying for my thesis, but since I did not use all the data I queried from Dune back then, I can just slim down my queries and save the new code in [dune_queries_rerun.sql](code/data/dune_queries_rerun.sql).

### Dune Query

Prices in DEX are updated with each new block that becomes indexed. Thus, to find the pool price at a given time, I need to look at the block that came before the given time. For example, if I need the price for the 2023-01-01 00:30:00 and I only have a block at 00:29:00 and 00:31:00, then the former holds the correct price information. As such, my query needs to give me block with the minimal timestamp smaller/equal to the desired timestamp. 

*For fun, I am trying to minimize the computation (I think that is how you would call it) of this query as far as possible. For that I looked into the order of execution in SQL. I am trying to minimize selecting and arithmetic before filtering. I don't know if this is even worth it, but it helps to understand SQL, which I haven't worked with since three years...*

The query now works. It calculates window_end differently than my other queries, because I wanted to try something new and they have the same operations. Choosing the block closest to window_end is done by row_number() over partition by window_end. 


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