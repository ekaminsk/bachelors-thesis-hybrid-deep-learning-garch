import time, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (UNISWAP_GRAPH_URL, UNISWAP_POOL_ID,                 #GraphQL url
                    UNISWAP_TICK_NUMBER, INTERVAL,                      #helpers
                    DEX_POOL, DEX_TICKS, DEX_MINTS_BURNS,  DEX_SWAPS,   #
                    UNISWAP_TOKEN0_DECIMAL_PLACES, UNISWAP_TOKEN1_DECIMAL_PLACES, 
                    UNISWAP_LARGE_TRADE_THRESHOLD,
                    UNISWAP_START_DATE, UNISWAP_END_DATE)
from utilities import gql, append_csv, sqrt_price_to_price
from datetime import datetime, timezone


# ── GraphQL query ───────────────────────────────────────────────────────────

QUERY = """
query DataCollection($pool_id: ID!, $block_nr: Int!, $timestamp_begin: BigInt!, $tickHigh: BigInt!, $tickLow: BigInt!){
  pool(
    id: $pool_id, 
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
}"""