"""
This file contains all the configurations for the following code. All fixed variables are in this file.
The division below is based on the use of the variable. All directory paths are relative.
"""
from dotenv import load_dotenv
import os

INTERVAL = 300                                                  # 5-min interval

#---API KEYS--------------------
load_dotenv()
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
UNISWAP_API_KEY = os.getenv("UNISWAP_API_KEY")

#---Directories-----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")                       # data directory
DEX_DIR = os.path.join(DATA_DIR, "DEX")                         # all the uniswap data
DEX_POOL = os.path.join(DEX_DIR, "pool")                        # pool data
DEX_MINTS_BURNS = os.path.join(DEX_DIR, "mints_burns")          # pool mints & burns
DEX_TICKS = os.path.join(DEX_DIR, "ticks")                      # pool ticks
DEX_SWAPS = os.path.join(DEX_DIR, "swaps_klines")               # pool swaps & klines from it
#-----
CEX_DIR = os.path.join(DATA_DIR, "CEX")                         # for data.vision.binance Klines and orderbook
CEX_KLINES = os.path.join(CEX_DIR, "klines")                    # only for downloaded klines
CEX_ORDERBOOK = os.path.join(CEX_DIR, "orderbook")              # local orderbook snapshots
#-----
DUNE_DIR = os.path.join(DATA_DIR, "ONCHAIN")                    # here all query responses can be fetched, because it is all in one file already
#-----
AGGREGATE_OUTPUT = os.path.join(DATA_DIR, "final")              # for the 5min aggregation
#-----
RESULTS_DIR = os.path.join(BASE_DIR, "MODEL_RESULTS")
TRAIN_RESULTS = os.path.join(RESULTS_DIR, "training_results.txt")
GARCH_OUTPUT = os.path.join(RESULTS_DIR, "garch_output.csv")
BEST_MODEL = os.path.join(RESULTS_DIR, "best_model.pt")        
MODEL_CONFIG = os.path.join(RESULTS_DIR, "model_config.json")  

#---Binance---------------------
BINANCE_SYMBOL = "usdcusdt"
BINANCE_ORDERBOOK_DEPTH = 20
BINANCE_SNAP_LIMIT = 1000
BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={BINANCE_SYMBOL.upper()}&limit={BINANCE_SNAP_LIMIT}"
BINANCE_WEBSOCKET = f"wss://stream.binance.com:9443/ws/{BINANCE_SYMBOL}@depth@100ms"

#---Uniswap---------------------
UNISWAP_SUBGRAPH_ID = (
    "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"              # Uniswap v3 ID
)    
UNISWAP_GRAPH_URL = (
    f"https://gateway.thegraph.com/api/{UNISWAP_API_KEY}"       # Accessing the Uniswap v3
    f"/subgraphs/id/{UNISWAP_SUBGRAPH_ID}"
    )
UNISWAP_POOL_ID = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"  # ID of the USDC/USDT pool
UNISWAP_TICK_NUMBER = 10                                        # [current_tick - TICK_NUMBER, current_tick + TICK_NUMBER]
UNISWAP_TOKEN0_DECIMAL_PLACES = 6                               # USDC
UNISWAP_TOKEN1_DECIMAL_PLACES = 6                               # USDT
UNISWAP_LARGE_TRADE_THRESHOLD = 100_000                         # What is considered a large trade
# I found the subgraph ID here: https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one

#---Dune------------------------
DUNE_START_DATE = "2026-01-01"
DUNE_END_DATE = "2026-01-31"
DUNE_WHALE_THRESHOLD = 1000000                                  # for query 1
DUNE_BASE_URL = "https://api.dune.com/api/v1"
DUNE_POLL_INTERVAL = 5                                          # seconds between status checks
DUNE_RESULTS_PER_PAGE = 10000                                   # rows per results page (max Dune allows)

#---Training--------------------
HIDDEN_LAYERS = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 6000
PATIENCE = 50                                                   # if the model has not improved after 50 epochs it stops
CLIP_NORM = 1.0                                                 # gradient clipping
ALPHA_SCALE_A = None                                            # scaling alpha
BETA_SCALE_B = None                                             # scaling beta

#---Evaluation------------------
PERMUTATION_REPETITIONS = 10                                    # permutation repetitions per feature (more = less noisy)
ACF_LAGS = 40                                                   # already used in the notebook too