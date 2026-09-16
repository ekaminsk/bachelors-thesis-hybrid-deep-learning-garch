-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 0 (ID: 8425803) Fetching Block Number for 5 min Intervals
-- ───────────────────────────────────────────────────────────────────────────

WITH FiveMinuteBlock AS (
    SELECT
        time                                                    AS real_time,
        (date_trunc('minute',time))
            - interval '1' minute * MOD(minute(time),5)
            + interval '5' minute                               AS window_end,
        number                                                  AS block_number
    FROM ethereum.blocks
    WHERE time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND time <  CAST('{{end_date}}' AS TIMESTAMP)
      AND date  >= CAST(CAST('{{start_date}}' AS TIMESTAMP) AS DATE)
      AND date  <  CAST(CAST('{{end_date}}' AS TIMESTAMP) AS DATE) + interval '1' day
),
TimeDifference AS (
    SELECT 
        window_end,
        real_time,
        block_number,
        row_number() OVER (
            PARTITION BY window_end
            ORDER BY date_diff('second',real_time,window_end) asc
        )                                                       AS row_count

    FROM FiveMinuteBlock
)
SELECT 
    window_end,
    real_time,
    block_number
FROM TimeDifference
WHERE row_count = 1
ORDER BY window_end asc;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 1 (ID: 6763552) Whale Transfers 
-- ───────────────────────────────────────────────────────────────────────────

WITH raw_transfers AS (
    SELECT
        (date_trunc('minute',t.block_time))
            - interval '1' minute * MOD(minute(t.block_time),5)
            + interval '5' minute                                     AS window_end,
        t.symbol                                                      AS token,
        t.amount_usd,
        CASE
            WHEN from_cex.cex_name IS NOT NULL AND to_cex.cex_name IS NULL THEN 'cex_outflow'
            WHEN from_cex.cex_name IS NULL     AND to_cex.cex_name IS NOT NULL THEN 'cex_inflow'
            WHEN from_cex.cex_name IS NOT NULL AND to_cex.cex_name IS NOT NULL THEN 'cex_to_cex'
            ELSE 'non_cex'
        END                                                           AS flow_direction
    FROM tokens.transfers t
    LEFT JOIN cex_ethereum.addresses from_cex ON t."from" = from_cex.address
    LEFT JOIN cex_ethereum.addresses to_cex   ON t."to"   = to_cex.address
    WHERE t.blockchain       = 'ethereum'
      AND t.symbol           IN ('USDC', 'USDT')            -- Here need to change, if I am using a different pair
      AND t.contract_address IN (
            0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48,     -- USDC smart contract
            0xdAC17F958D2ee523a2206206994597C13D831ec7      -- USDT smart contract
          )
      AND t.amount_usd       >= {{whale_threshold_usd}}
      AND t.block_time       >= CAST('{{start_date}}' AS TIMESTAMP)
      AND t.block_time       <  CAST('{{end_date}}'   AS TIMESTAMP)
)

SELECT
    window_end,
    token,
    flow_direction,
    COUNT(*)                          AS transfer_count,
    SUM(amount_usd)                   AS total_usd,
FROM raw_transfers
GROUP BY 1, 2, 3
ORDER BY window_end ASC, token, flow_direction;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 2 (ID: 6763555) CEX Inflows and Outflows 
-- ───────────────────────────────────────────────────────────────────────────

WITH raw_data AS(
    SELECT
        block_time,
        date_trunc('minute', block_time)
            - interval '1' minute * MOD(minute(block_time), 5)
            + interval '5' minute             AS window_end,
        amount_usd,
        token_symbol,
        flow_type
    FROM cex.flows
    WHERE
        blockchain = 'ethereum'
    AND (token_address = 0xdac17f958d2ee523a2206206994597c13d831ec7 
        OR token_address = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48) 
        AND block_time   >= CAST('{{start_date}}' AS TIMESTAMP)
        AND block_time   <  CAST('{{end_date}}'   AS TIMESTAMP)
)

SELECT
    window_end,
    token_symbol,
    flow_type,
    SUM(amount_usd)   AS total_usd

FROM raw_data
WHERE flow_type IN ('Inflow', 'Outflow')
GROUP BY 1, 2, 3
ORDER BY window_end asc, token_symbol, total_usd desc;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 3 (ID: 6763557) Gas Price Time Series (Base + Priority Fee) 
-- ───────────────────────────────────────────────────────────────────────────

WITH raw_gas_data AS(
    SELECT
        date_trunc('minute', time)
            - interval '1' minute * MOD(minute(time), 5)
            + interval '5' minute             AS window_end,
        AVG(gas_used)/1e9                     AS avg_base_fee_gwei
    FROM ethereum.blocks
    WHERE time >= CAST('{{start_date}}' AS TIMESTAMP)
        AND time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),
transaction_priority_fee AS(
    SELECT
        date_trunc('minute', block_time)
            - interval '1' minute * MOD(minute(block_time), 5)
            + interval '5' minute             AS window_end,
        
        APPROX_PERCENTILE(
            CASE WHEN type = 'DynamicFee' 
                THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.5)
                AS priority_fee_p50_gwei,
        
        APPROX_PERCENTILE(
            CASE WHEN type = 'DynamicFee' 
                THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.8)
                AS priority_fee_p80_gwei
    
    FROM ethereum.transactions
    WHERE block_time >= CAST('{{start_date}}' AS TIMESTAMP)
        AND block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
    ORDER BY window_end asc
)

SELECT 
    r.window_end,
    r.avg_base_fee_gwei,
    t.priority_fee_p50_gwei,
    t.priority_fee_p80_gwei,
    r.avg_base_fee_gwei + COALESCE(t.priority_fee_p50_gwei,0) 
        AS approx_effective_gas_gwei
FROM raw_gas_data r
LEFT JOIN transaction_priority_fee t ON r.window_end = t.window_end
ORDER BY r.window_end asc;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 4 (ID: 6763559) Gas Used Per Block & Block Utilization 
-- ───────────────────────────────────────────────────────────────────────────

SELECT 
    date_trunc('minute', time)
        - interval '1' minute * MOD(minute(time), 5)
        + interval '5' minute             AS window_end,
    AVG(CAST(gas_used as double) / CAST(gas_limit as double))   
        AS avg_utilization,
    CAST(
        SUM(CASE WHEN CAST(gas_used AS double) / CAST(gas_limit AS double) > 0.8
            THEN 1 ELSE 0 END) AS double
    ) / COUNT(*)                                                    
        AS pct_blocks_near_full
FROM ethereum.blocks
WHERE time >= CAST('{{start_date}}' AS TIMESTAMP)
    AND time <  CAST('{{end_date}}'   AS TIMESTAMP)
GROUP BY 1
ORDER BY window_end asc;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 5 (ID: 6763560) Mempool Congestion Proxies
-- ───────────────────────────────────────────────────────────────────────────

WITH block_metrics AS (
    SELECT
        date_trunc('minute', time)
            - interval '1' minute * MOD(minute(time), 5)
            + interval '5' minute           AS window_end,
        AVG(CAST(base_fee_per_gas / 1e9 AS double))
            AS avg_base_fee_gwei,
        AVG(CAST(gas_used AS double) / CAST(gas_limit AS double))   
            AS avg_fill_ratio  
    FROM ethereum.blocks
    WHERE time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),
fee_metrics AS(
    SELECT 
        date_trunc('minute', block_time)
            - interval '1' minute * MOD(minute(block_time), 5)
            + interval '5' minute           AS window_end,
        APPROX_PERCENTILE(
            CASE WHEN type = 'DynamicFee' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.8
        ) - APPROX_PERCENTILE(
            CASE WHEN type = 'DynamicFee' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.1
        )                                                           
            AS priority_fee_spread_gwei
    FROM ethereum.transactions
    WHERE block_time >= CAST('{{start_date}}' AS TIMESTAMP)
        AND block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),
combined AS(
    SELECT
        b.window_end,
        b.avg_base_fee_gwei,
        b.avg_fill_ratio,
        f.priority_fee_spread_gwei,
        
        (b.avg_base_fee_gwei - LAG(b.avg_base_fee_gwei, 1) OVER (ORDER BY b.window_end))
        / NULLIF(LAG(b.avg_base_fee_gwei, 1) OVER (ORDER BY b.window_end), 0)
            AS base_fee_pct_change

    FROM block_metrics b
    LEFT JOIN fee_metrics f ON b.window_end = f.window_end
)
SELECT
    window_end,
    base_fee_pct_change,
    LEAST(1.0,
        avg_fill_ratio * 0.4
        + CASE WHEN base_fee_pct_change > 0 THEN 0.3 ELSE 0.0 END
        + LEAST(COALESCE(priority_fee_spread_gwei, 0), 20.0) / 20.0 * 0.3
    )     
        AS congestion_score
FROM combined
ORDER BY window_end asc;


-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 6 (ID: 6763561) On-Chain Mints and Burns
-- ───────────────────────────────────────────────────────────────────────────

WITH usdc_mints AS (
    SELECT
        evt_block_time  AS event_time,
        'USDC'          AS token,
        'mint'          AS event_type,
        CAST(amount AS double) / 1e6  AS token_amount
    FROM circle_ethereum.usdc_evt_mint
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND evt_block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
),

usdc_burns AS (
    SELECT
        evt_block_time,
        'USDC',
        'burn',
        CAST(amount AS double) / 1e6
    FROM circle_ethereum.usdc_evt_burn
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND evt_block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
),

usdt_mints AS (
    SELECT
        evt_block_time,
        'USDT',
        'mint',
        CAST(amount AS double) / 1e6
    FROM tether_ethereum.tether_usd_evt_issue
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND evt_block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
),

usdt_burns AS (
    SELECT
        evt_block_time,
        'USDT',
        'burn',
        CAST(amount AS double) / 1e6
    FROM tether_ethereum.tether_usd_evt_redeem
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND evt_block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
),

usdt_burn_blacklist AS (
    SELECT
        evt_block_time,
        'USDT',
        'burn_blacklist',
        CAST(_balance AS double) / 1e6
    FROM tether_ethereum.tether_usd_evt_destroyedblackfunds
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND evt_block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
),

all_events AS (
    SELECT * FROM usdc_mints
    UNION ALL SELECT * FROM usdc_burns
    UNION ALL SELECT * FROM usdt_mints
    UNION ALL SELECT * FROM usdt_burns
    UNION ALL SELECT * FROM usdt_burn_blacklist
),

agg_5min AS (
    SELECT
        date_trunc('minute', event_time)
            - interval '1' minute * MOD(minute(event_time), 5)
            + interval '5' minute           AS window_end,
        token,
        event_type,
        SUM(token_amount)                   AS total_token_amount
    FROM all_events
    GROUP BY 1, 2, 3
)

SELECT
    window_end,
    token,
    event_type,
    total_token_amount
FROM agg_5min
ORDER BY window_end ASC, token, event_type;