-- ============================================================
-- QUERY 1 (ID: 6763552)
-- ============================================================

WITH raw_transfers AS (
    SELECT
        -- 5-min window_end: event at 00:03:47 -> 00:05:00
        date_trunc('hour', t.block_time)
            + interval '5' minute * (minute(t.block_time) / 5 + 1)  AS window_end,
        t.symbol                                                      AS token,
        t.amount,
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
      AND t.symbol           IN ('USDC', 'USDT')
      AND t.contract_address IN (
            0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48,
            0xdAC17F958D2ee523a2206206994597C13D831ec7
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
    SUM(amount)                       AS total_token_amount,
    SUM(amount_usd)                   AS total_usd,
    AVG(amount_usd)                   AS avg_usd_per_transfer,
    MIN(amount_usd)                   AS min_usd_per_transfer,
    MAX(amount_usd)                   AS max_usd_per_transfer,
    APPROX_PERCENTILE(amount_usd, 0.5) AS median_usd_per_transfer
FROM raw_transfers
GROUP BY 1, 2, 3
ORDER BY window_end ASC, token, flow_direction;


-- ============================================================
-- QUERY 2 (ID: 6763555)
-- ============================================================

WITH raw_flows AS (
    SELECT
        date_trunc('hour', block_time)
            + interval '5' minute * (minute(block_time) / 5 + 1)  AS window_end,
        cex_name,
        token_symbol,
        flow_type,
        amount_usd,
        amount,
        tx_hash
    FROM cex.flows
    WHERE blockchain    = 'ethereum'
      AND token_symbol  IN ('USDC', 'USDT')
      AND token_address IN (
            0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48,
            0xdAC17F958D2ee523a2206206994597C13D831ec7
          )
      AND block_time   >= CAST('{{start_date}}' AS TIMESTAMP)
      AND block_time   <  CAST('{{end_date}}'   AS TIMESTAMP)
)

SELECT
    window_end,
    cex_name,
    token_symbol,
    flow_type,
    COUNT(*)                                        AS transfer_count,
    SUM(amount)                                     AS total_token_amount,
    SUM(amount_usd)                                 AS total_usd,
    AVG(amount_usd)                                 AS avg_usd_per_transfer,
    APPROX_PERCENTILE(amount_usd, 0.5)              AS median_usd_per_transfer,
    APPROX_PERCENTILE(amount_usd, 0.9)              AS p90_usd_per_transfer,
    SUM(CASE WHEN flow_type = 'inflow'  THEN  amount_usd
             WHEN flow_type = 'outflow' THEN -amount_usd
             ELSE 0 END)                            AS net_usd_flow
FROM raw_flows
GROUP BY 1, 2, 3, 4
ORDER BY window_end ASC, total_usd DESC;


-- ============================================================
-- QUERY 3 (ID: 6763557)
-- ============================================================

WITH block_base_fee AS (
    SELECT
        date_trunc('hour', time)
            + interval '5' minute * (minute(time) / 5 + 1)         AS window_end,
        COUNT(*)                                                    AS block_count,
        AVG(base_fee_per_gas) / 1e9                                 AS avg_base_fee_gwei,
        MIN(base_fee_per_gas) / 1e9                                 AS min_base_fee_gwei,
        MAX(base_fee_per_gas) / 1e9                                 AS max_base_fee_gwei,
        APPROX_PERCENTILE(CAST(base_fee_per_gas AS double), 0.5) / 1e9
                                                                    AS median_base_fee_gwei
    FROM ethereum.blocks
    WHERE date >= CAST('{{start_date}}' AS TIMESTAMP)
      AND date <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),
tx_priority_fee AS (
    SELECT
        date_trunc('hour', block_time)
            + interval '5' minute * (minute(block_time) / 5 + 1)   AS window_end,
        COUNT(*)                                                    AS tx_count,
        APPROX_PERCENTILE(CAST(gas_price AS double) / 1e9, 0.1)    AS gas_price_p10_gwei,
        APPROX_PERCENTILE(CAST(gas_price AS double) / 1e9, 0.5)    AS gas_price_p50_gwei,
        APPROX_PERCENTILE(CAST(gas_price AS double) / 1e9, 0.8)    AS gas_price_p80_gwei,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.1
        )                                                           AS priority_fee_p10_gwei,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.5
        )                                                           AS priority_fee_p50_gwei,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.8
        )                                                           AS priority_fee_p80_gwei
    FROM ethereum.transactions
    WHERE block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
)

SELECT
    b.window_end,
    b.block_count,
    b.avg_base_fee_gwei,
    b.min_base_fee_gwei,
    b.max_base_fee_gwei,
    b.median_base_fee_gwei,
    t.tx_count,
    t.gas_price_p10_gwei,
    t.gas_price_p50_gwei,
    t.gas_price_p80_gwei,
    t.priority_fee_p10_gwei,
    t.priority_fee_p50_gwei,
    t.priority_fee_p80_gwei,
    b.avg_base_fee_gwei + COALESCE(t.priority_fee_p50_gwei, 0)     AS approx_effective_gas_gwei
FROM block_base_fee b
LEFT JOIN tx_priority_fee t ON b.window_end = t.window_end
ORDER BY b.window_end ASC;


-- ============================================================
-- QUERY 4 (ID: 6763559)
-- ============================================================

SELECT
    date_trunc('hour', time)
        + interval '5' minute * (minute(time) / 5 + 1)             AS window_end,
    COUNT(*)                                                        AS block_count,
    AVG(CAST(gas_used  AS double))                                  AS avg_gas_used,
    MIN(CAST(gas_used  AS double))                                  AS min_gas_used,
    MAX(CAST(gas_used  AS double))                                  AS max_gas_used,
    APPROX_PERCENTILE(CAST(gas_used AS double), 0.5)                AS median_gas_used,
    AVG(CAST(gas_limit AS double))                                  AS avg_gas_limit,
    AVG(CAST(gas_used AS double) / CAST(gas_limit AS double))       AS avg_utilization,
    APPROX_PERCENTILE(
        CAST(gas_used AS double) / CAST(gas_limit AS double), 0.5
    )                                                               AS median_utilization,
    MAX(CAST(gas_used AS double) / CAST(gas_limit AS double))       AS max_utilization,
    CAST(
        SUM(CASE WHEN CAST(gas_used AS double) / CAST(gas_limit AS double) > 0.5
                 THEN 1 ELSE 0 END) AS double
    ) / COUNT(*)                                                    AS pct_blocks_above_target,
    CAST(
        SUM(CASE WHEN CAST(gas_used AS double) / CAST(gas_limit AS double) > 0.8
                 THEN 1 ELSE 0 END) AS double
    ) / COUNT(*)                                                    AS pct_blocks_near_full,
    AVG(base_fee_per_gas) / 1e9                                     AS avg_base_fee_gwei,
    AVG(CAST(blob_gas_used AS double))                              AS avg_blob_gas_used,
    AVG(CAST(size AS double))                                       AS avg_block_size_bytes
FROM ethereum.blocks
WHERE date >= CAST('{{start_date}}' AS TIMESTAMP)
  AND date <  CAST('{{end_date}}'   AS TIMESTAMP)
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- QUERY 5 (ID: 6763560)
-- ============================================================

WITH block_metrics AS (
    SELECT
        date_trunc('hour', time)
            + interval '5' minute * (minute(time) / 5 + 1)         AS window_end,
        COUNT(*)                                                    AS block_count,
        AVG(CAST(gas_used AS double) / CAST(gas_limit AS double))   AS avg_fill_ratio,
        MAX(CAST(gas_used AS double) / CAST(gas_limit AS double))   AS max_fill_ratio,
        AVG(base_fee_per_gas / 1e9)                                 AS avg_base_fee_gwei,
        CAST(
            SUM(CASE WHEN CAST(gas_used AS double) / CAST(gas_limit AS double) > 0.8
                     THEN 1 ELSE 0 END) AS double
        ) / COUNT(*)                                                AS pct_blocks_near_full,
        CAST(
            SUM(CASE WHEN CAST(gas_used AS double) / CAST(gas_limit AS double) > 0.5
                     THEN 1 ELSE 0 END) AS double
        ) / COUNT(*)                                                AS pct_blocks_above_target
    FROM ethereum.blocks
    WHERE date >= CAST('{{start_date}}' AS TIMESTAMP)
      AND date <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),

fee_metrics AS (
    SELECT
        date_trunc('hour', block_time)
            + interval '5' minute * (minute(block_time) / 5 + 1)   AS window_end,
        COUNT(*)                                                    AS tx_count,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.8
        ) - APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.1
        )                                                           AS priority_fee_spread_gwei,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.5
        )                                                           AS priority_fee_median_gwei,
        APPROX_PERCENTILE(
            CASE WHEN type = '2' THEN CAST(priority_fee_per_gas AS double) / 1e9 END, 0.8
        )                                                           AS priority_fee_p80_gwei,
        AVG(CAST(gas_limit AS double))                              AS avg_tx_gas_limit,
        AVG(CAST(gas_used  AS double))                              AS avg_tx_gas_used
    FROM ethereum.transactions
    WHERE block_time >= CAST('{{start_date}}' AS TIMESTAMP)
      AND block_time <  CAST('{{end_date}}'   AS TIMESTAMP)
    GROUP BY 1
),

combined AS (
    SELECT
        b.window_end,
        b.block_count,
        b.avg_fill_ratio,
        b.max_fill_ratio,
        b.avg_base_fee_gwei,
        b.pct_blocks_near_full,
        b.pct_blocks_above_target,
        -- LAG now compares adjacent 5-min windows (5-min momentum signal)
        (b.avg_base_fee_gwei - LAG(b.avg_base_fee_gwei, 1) OVER (ORDER BY b.window_end))
        / NULLIF(LAG(b.avg_base_fee_gwei, 1) OVER (ORDER BY b.window_end), 0)
                                                                    AS base_fee_pct_change,
        f.tx_count,
        f.priority_fee_spread_gwei,
        f.priority_fee_median_gwei,
        f.priority_fee_p80_gwei,
        f.avg_tx_gas_limit,
        f.avg_tx_gas_used
    FROM block_metrics b
    LEFT JOIN fee_metrics f USING (window_end)
)

SELECT
    window_end,
    block_count,
    tx_count,
    avg_fill_ratio,
    max_fill_ratio,
    pct_blocks_near_full,
    pct_blocks_above_target,
    avg_base_fee_gwei,
    base_fee_pct_change,
    priority_fee_median_gwei,
    priority_fee_p80_gwei,
    priority_fee_spread_gwei,
    avg_tx_gas_limit,
    avg_tx_gas_used,
    LEAST(1.0,
        avg_fill_ratio * 0.4
        + CASE WHEN base_fee_pct_change > 0 THEN 0.3 ELSE 0.0 END
        + LEAST(COALESCE(priority_fee_spread_gwei, 0), 20.0) / 20.0 * 0.3
    )                                                               AS congestion_score
FROM combined
ORDER BY window_end;


-- ============================================================
-- QUERY 6 (ID: 6763561)
-- ============================================================

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
    -- Forced destruction of blacklisted wallets -- distinct supply reduction event
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
        date_trunc('hour', event_time)
            + interval '5' minute * (minute(event_time) / 5 + 1)   AS window_end,
        token,
        event_type,
        COUNT(*)                                                    AS event_count,
        SUM(token_amount)                                           AS total_token_amount,
        MIN(token_amount)                                           AS min_single_amount,
        MAX(token_amount)                                           AS max_single_amount,
        AVG(token_amount)                                           AS avg_amount
    FROM all_events
    GROUP BY 1, 2, 3
)

SELECT
    window_end,
    token,
    event_type,
    event_count,
    total_token_amount,
    min_single_amount,
    max_single_amount,
    avg_amount,
    SUM(
        CASE WHEN event_type = 'mint' THEN total_token_amount ELSE -total_token_amount END
    ) OVER (
        PARTITION BY token
        ORDER BY window_end
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                               AS cumulative_net_supply_delta
FROM agg_5min
ORDER BY window_end ASC, token, event_type;
