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