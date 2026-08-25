-- ───────────────────────────────────────────────────────────────────────────
-- QUERY 7 (ID: 8425803) Fetching Block Number for 5 min Intervals
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
ORDER BY window_end asc
