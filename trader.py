import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import pandas as pd
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    position_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    starfruit_prices = pd.Series()

    # input: OrderDepth order_depth
    # return: mid_price
    # calculates mid price of all orders in an OrderDepth
    # calculates average of all asks and bids, then returns the median of the two
    def calculate_mid_price(self, order_depth: OrderDepth):
        avg_bid_price = 0
        buy_counter = 0
        avg_ask_price = 0
        sell_counter = 0

        for value in order_depth.buy_orders.values():
            avg_bid_price += value
            buy_counter += 1

        avg_bid_price /= buy_counter

        for value in order_depth.sell_orders.values():
            avg_ask_price += value
            sell_counter += 1

        sell_counter /= sell_counter

        return (avg_ask_price + avg_bid_price) / 2
    
    # input: 
    # prices: pd.Series - mid_price for each timestamp for a given product
    # window: int - the window for the moving average
    # return: pd.Series with appended column that has the moving average
    # calculate moving average of a pandas series and returns a dataframe with a new column that has the new moving average
    def calculate_moving_average(self, prices: pd.Series, window: int):
        if len(prices) < window:
            return None
        else:
            prices[f"mid_price_ma{window}"] = prices['mid_price'].rolling(window=window).mean()
            prices.fillna(0, inplace=True)
            return prices

    # so far
    # extracts features for starfruit for linear regression
    def extract_features(self, state: TradingState, symbol: str):
        trades = state.order_depths[symbol]
        mid_price = self.calculate_mid_price(trades)
        self.starfruit_prices.append(mid_price, inplace=True, index=state.timestamp)

        if len(starfruit_prices > 10):
            starfruit_prices = self.calculate_moving_average(starfruit_prices, 10)

    def predict_prices(features: pd.DataFrame, coefficients: int, intercept: int):
        return np.dot(features, coefficients) + intercept

    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        
        for product in state.order_depths:

            if product == "STARFRUIT":
                coefficients = [1, 8.35469697e-17]
                intercept = -2.091837814077735e-11
                threshold = 5

                predicted_price = self.predict_prices(self.starfruit_prices, coefficients, intercept)
                acceptable_price_range = [predicted_price-threshold, predicted_price+threshold]

                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        if best_ask < acceptable_price[1]:
                            buy_amount = min(self.position_limit[product] - self.positions[product], -best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product, best_ask, best_ask_amount))
                            self.positions[product] += buy_amount
                            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]

                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if best_bid > acceptable_price_range[0] and self.positions[product] > -self.position_limit[product]:
                        sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                        logger.print("SELL", str(sell_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_amount))
                        self.positions[product] -= sell_amount
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]

            
            if product == 'AMETHYSTS':  
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                acceptable_price = 10000
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limit[product] - self.positions[product],-best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                    
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limit[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                            logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            
                result[product] = orders
                logger.print('\n'+str(self.positions[product]))
                #print(state.position[product])
                
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData