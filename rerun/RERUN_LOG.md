# Hybrid Deep Learning GARCH for Stablecoin Volatility Forecasting - Model Rerun

This is the log for rerunning this project. I will keep updates in here before committing to organizing everything (potentially) under a new folder, new notebooks, a second paper, etc. This file is ordered in reverse chronological order, such that new updates are at the top.

## 05.09.2026 - Continuing with uniswap data collection 

Currently, running the query returns the following dictionary:

```
{
  "data": {
    "pool": {
      "burns": [
        "timestamp",
        "amount"
      ],
      "liquidity",
      "mints": [
        "timestamp",
        "amount"
      ],
      "sqrtprice",
      "swaps": [
        "amount0",
        "amountUSD",
        "timestamp"
      ]
      "tick"
      "ticks": [
        "liquidityGross",
        "liquidityNet",
        "tickIdx"
      ],
      "totalvaluelocked"
    }
  }
}
```
Now there is an issue I should fix immediately. As far as I understand, the subgraph I am querying is indexed by different users. This means that the data I have available for querying depends on the indexer I pull when sending my query. From testing, I can conclude that there is at least one indexer, who has indexed far enough back to query data from 2023, but there is also an indexer who has pruned away the blocks I am time-traveling to. Simple fix is to add an 'if error' clause. 


## 28.08.2026 - Building univ3_pool_historical.py

First, I need to collect what I need this program to do. On a meta-level I need it to take a GraphQL query, connect to theGraph and run the query; take the result of said query and output it in CSV. More granularly: 
- Take three GraphQL queries with fixed pool address (and low/high for ticks)
- Take the block_number table at 5-minute intervals
- Run each query for each 5-min interval (-> here I may run into credit constraints...)
- sqrtPrice to Price
- Either immediately append to a CSV or keep locally for a while, then add to CSV
Further cleaning happens in aggregate_5min_rerun.py.

While looking through GraphQL documentation and my rerun-aggregation, I am realizing a few things. 
- First, by removing the unnecessary metrics from my aggregate_5min_rerun.py, I also need to remove these from the collection, since renaming the fields goes in order (i.e., if I have Klines with open, close, high, and low and those are aggregated to dex_open, dex_close, dex_high, and dex_low, then dropping metrics to only dex_open, dex_high now maps: open -> dex_open, close -> dex_high, high -> NULL, low -> NULL). 
- Second, I can most likely summarize all three queries into one query. In the [GraphQL schema](https://github.com/Uniswap/v3-subgraph/blob/main/src/v3/schema.graphql) for Uniswap V3, mints, burns, and ticks all are derivable from pool. But I need to try that out.

The first thing I need to do is clean out which fields I actually need. Klines are much more annoying, since they are derived from raw swaps, so I would need to rebuild that too, if I want to clean up my rerun:
```
dex_price                           <->       sqrtPrice               *direct pool query*

dex_pool_liquidity                  <->       liquidity               *direct pool query*
dex_pool_tvl_usd                    <->       totalValueLockedUSD     *direct pool query*
current_tick                        <->       tick                    *direct pool query*

dex_ticks_total_liq_gross           <->       liquidityGross          *derived from tick query (sum over all ticks)*
dex_ticks_net_liq_above             <->       liquidityNet & tickIdx  *derived as sum(liqNet) if current_tick < tickIdx
dex_ticks_net_liq_below             <->       liquidityNet & tickIdx  *derived as sum(liqNet) if current_tick > tickIdx
dex_ticks_n_active                  <->       len(ticks)              *derived as #entries in query

dex_lp_net_liq_change               <->       amount                  *derived from burn/mint query*
dex_lp_n_mints                      <->       len(mints)              *derived from burn/mint query*
dex_lp_n_burns                      <->       len(burns)              *derived from burn/mint query*

dex_klines_volume_usd               <->       volume_usd              *derived from swap query*
dex_klines_n_swaps                  <->       n_swaps                 *derived from swap query*
dex_klines_imbalance                <->       imbalance               *derived from swap query*
dex_klines_large_trades_count       <->       large_trades_count      *derived from swap query*
dex_klines_large_trades_usd         <->       large_trades_usd        *derived from swap query*
```

Turns out, this query is buildable, and I can take it further. Swaps also can be reverse-looked-up through pool, meaning that I can write one query combining the entire data collection:

```
query DataCollection($pool_id: ID!, $block_nr: Int!, $timestamp_begin: BigInt!, $tickHigh: BigInt!, $tickLow: BigInt!){
  pool(id: $pool_id, 
    block: {number: $block_nr}){
    sqrtPrice,
    totalValueLockedUSD,
    liquidity,
    tick,
    ticks(
      first: 100
    	where: {tickIdx_gte: $tickLow, tickIdx_lte: $tickHigh}
      orderBy: tickIdx
      orderDirection: asc
    ){     
      liquidityGross,
      liquidityNet,
      tickIdx
    },
    mints(
      first: 100
      where: {timestamp_gte: $timestamp_begin}
      orderBy: timestamp
      orderDirection:desc
    ){
      timestamp,
      amount
    }
  	burns(
      first: 100
      where: {timestamp_gte: $timestamp_begin}
      orderBy: timestamp
      orderDirection:desc
    ){
      timestamp,
      amount
    }
    swaps(  
      first: 100
      where: {timestamp_gte: $timestamp_begin}
      orderBy: timestamp
      orderDirection: desc
    ){
      amountUSD,
      timestamp,
      amount0    
    }
  }
}
```

This warrants renaming the univ3_pool_historical.py to univ3_rerun.py

## 27.08.2026 - Finishing the Dune Query rebuild

Opening Dune I am prompted with the message that starting September 10th, the free plan, which I have been using for this thesis, will no longer work as before. Instead, I will have a 14-day trial for the plus plan. As far as I understand, creating new queries and running any queries and especially using an API to capture the data locally is going to now cost $349 per month. Thus, this rerun now has a strict deadline (24th of September) as I need Dune queries for both my DEX data (due to block_number for historical queries) and on-chain metrics.

**Query 6:** Removed the avg/min/max metrics and net_cumulative_supply

With that, I have finished retouching all the queries. Next I need to rebuild the DEX data collection queries and code, adding the block parameter for historical retrieval. 


## 26.08.2026 - Continuing the Dune Query rebuild

### Aggregation code

Before building the next query, I am realizing that I also can rebuild the aggeregate_5min.py, as I am not using some of the data outputted by the aggregation step. 

- **CEX:** First I remove all the CEX loaders, since I did not use Kline data in my model anyway and I am not going to collect orderbook data due to time constraints. 
- **DEX Kline:** For DEX Kline data I only used volume-based inputs (volume, imbalance, large trades), so I am removing all price-related metrics. 
- **DEX Swaps:** I also did not use any of the dex_swap_... metrics.
- **DEX Pool:** I only used dex_pool_liquidity and dex_pool_tvl_usd 
- **DEX Ticks:** Only keeping dex_ticks_total_liq_gross, dex_ticks_net_liq_above, dex_ticks_net_liq_below, and dex_ticks_n_active
- **DEX Liquidityprovider:** Only keeping dex_lp_net_liq_change, dex_lp_n_mints, and dex_lp_n_burns
- **Dune Whale transfers:** Removing the transfer_counts from pivot
- **Dune CEX flows:** Removing transfer_counts from pivot
- **Dune Gas:** Only keeping dune_gas_base_fee_gwei, dune_gas_tip_p50_gwei,
dune_gas_tip_p80_gwei and dune_gas_effective_gwei
- **Dune mempool:** I collected so much yet am only using two metrics, the dune_mempool_congestion_score and dune_mempool_base_fee_change
- **Dune block:** Only using dune_block_utilization and dune_block_pct_near_full
- **Dune supply changes:** Only using total_token_amounts

Overall, this cuts down the aggregation step drastically. 

### Dune queries
Now that I can better tell what I actually need in data, it is easier to rewrite the Dune queries.

**Query 2:** Cut down most metrics (avg/max/min/medium), net_flows, and transaction_count, as I am not using them in my model. Also removed unnecessary filtration; there is no need to filter for both USDT/USDC and their unique smart contract ID, especially after filtering for ethereum.

**Query 3:** Aside from removing all the metrics that I am not using, I ran into a monstrous bug: The Ethereum blockchain has different transaction types, one that has a priority fee / tip component in their gas (EIP 1559) and other types, which do not have that split fee. When I built this query, I connected Claude through an MCP to Dune, because searching through their tables was too much work. And Claude built the condition
> CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END
Turns out, there is no type = '2' in that table. What I would have needed was type = 'DynamicFee,' so naturally in my collection, each entry had priority_fee_per_gas as a NULL. Again, it didn't change much with the lack of ARCH effects, but it's a good thing I am catching this.

**Query 4:** Very simple, just removed everything aside from avg_utilization and pct_blocks_near_full.

**Query 5:** Similar to query 3; remove unneccesary metrics & change type = '2' to type = 'DynamicFee'.


## 25.08.2026 - Designing historical data collection system

Immediately, I thnik that I need a Dune query to output the corresponding block to each 5min interval aligned to UTC clock. This will then be inputted into a new [univ3_pool_historical.py](code/data/univ3_rerun.py). The issue now is that I am running into the same Dune credit constraint I had while querying for my thesis, but since I did not use all the data I queried from Dune back then, I can just slim down my queries and save the new code in [dune_queries_rerun.sql](code/data/dune_queries_rerun.sql).

### Dune Query 0 - Getting Historical Block Numbers

Prices in DEX are updated with each new block that becomes indexed. Thus, to find the pool price at a given time, I need to look at the block that came before the given time. For example, if I need the price for the 2023-01-01 00:30:00 and I only have a block at 00:29:00 and 00:31:00, then the former holds the correct price information. As such, my query needs to give me block with the minimal timestamp smaller/equal to the desired timestamp. 

*For fun, I am trying to minimize the computation (I think that is how you would call it) of this query as far as possible. For that I looked into the order of execution in SQL. I am trying to minimize selecting and arithmetic before filtering. I don't know if this is even worth it, but it helps to understand SQL, which I haven't worked with since three years...*

The query now works. It calculates window_end differently than my other queries, because I wanted to try something new and they have the same operations. Choosing the block closest to window_end is done by row_number() over partition by window_end. 

### Rebuilding Dune Queries 1-6

One thing that I have realized is that I am not acutally using most of the data that I collect from Dune. Therefore it would make sense to just cut it all. Especially now that I will collect a much, much longer period and add the block query. All that I need to do here is to check what I am aggergating in aggregate_5min.py and remove the rest.

**Query 1:** I am removing total_token_amount and the avg/min/max/median_usd_per_transfer.

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