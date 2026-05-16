# API KEYS
DUNE_API_KEY
UNISWAP_API_KEY

# DIRECTORIES
## Data
DEX_DIR - for any DEX data (swaps, pool)
    DEX_POOL
    DEX_MINTS_BURNS
    DEX_TICKS
    DEX_SWAPS
CEX_DIR - for data.vision.binance Klines and orderbook
    CEX_KLINES - only for downloaded klines
    CEX_ORDERBOOK
DUNE_DIR - here all query responses can be fetched, because it is all in one file already
AGGERGATE_OUTPUT - for the aggregate_5min function later

## Evaluation
RESULTS_DIR
    TRAIN_RESULTS
    GARCH_OUTPUT
    BEST_MODEL - .pt file
    MODEL_CONFIG - .json file
    RESULTS_TXT

## Data
DATA_DIR


# QUERY_STUFF
## Overarching
INTERVAL = 300

## Binance
BINANCE_SYMBOL = usdcusdt
BINANCE_ORDERBOOK_DEPTH = 20
BINANCE_SNAP_LIMIT = 1000
BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={BINANCE_SYMBOL.upper()}&limit={BINANCE_SNAP_LIMIT}"
BINANCE_WEBSOCKET = f"wss://stream.binance.com:9443/ws/{BINANCE_SYMBOL}@depth@100ms"

## Uniswap 
UNISWAP_SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
UNISWAP_GRAPH_URL = f"https://gateway.thegraph.com/api/{UNISWAP_API_KEY}/subgraphs/id/{UNISWAP_SUBGRAPH_ID}"
UNISWAP_POOL_ID = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"
UNISWAP_TICK_NUMBER = 10
UNISWAP_TOKEN0_DECIMAL_PLACES = 6 - here usdc
UNISWAP_TOKEN1_DECIMAL_PLACES = 6 - here usdt
UNISWAP_LARGE_TRADE_THRESHOLD = 100_000

I found the subgraph ID here: https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one for next 

## Dune
DUNE_START_DATE
DUNE_END_DATE
DUNE_WHALE_THRESHOLD - for query 1
DUNE_BASE_URL = "https://api.dune.com/api/v1"
DUNE_POLL_INTERVAL = 5
DUNE_RESULTS_PER_PAGE = 10000 - max Dune allows

# EVALUATION_STUFF
PERMUTATION_REPETITIONS = 10 - permutation repetitions per feature (more = less noisy)
ACF_LAGS = 40 - already used in the notebook too
DEVICE = torch.device("cpu") - sth for torch

# TRAINING_STUFF
HIDDEN_LAYERS = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 6000
PATIENCE = 50 - idk what that is
CLIP_NORM = 1.0 - idk what that is
ALPHA_SCALE_A
BETA_SCALE_B