import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order
import collections
import random
import math
import copy
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
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}
    position_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    starfruit_mid_price_log = {'STARFRUIT': []}
    starfruit_price_log = {'STARFRUIT':[]}
    basket_price_log = {'COMBINED': [], 'BASKET': [], 'DELTA': []}
    trending = 0    

    def moving_average(self, item, period):
        '''
        Calculate the weighted moving average price for the specified item over the specified period.
        '''

        # Check if the item exists in the starfruit_mid_price_log dictionary 
        # and has enough entries to calculate the moving average for the given period.
        if item in self.starfruit_mid_price_log and len(self.starfruit_mid_price_log[item]) >= period:

            # Extract the last 'period' number of prices for the specified item.
            prices = self.starfruit_mid_price_log[item][-period:]
            
            # Generate a range of weights, it is linear in this case.
            weights = range(1, period + 1)
            
            # Calculate the weighted sum of the prices. This is done by multiplying each price by its corresponding weight
            # and then summing up the results. This step emphasizes more recent prices by assigning them higher weights.
            weighted_sum = sum(price * weight for price, weight in zip(prices, weights))
            
            # Calculate the total weight by summing up all the weights.
            total_weight = sum(weights)
            
            # Calculate and return the weighted average price. This is the weighted sum divided by the total weight.
            # This results in an average where more recent prices have a greater influence than older prices.
            #return weighted_sum / total_weight
            coef = [0.198477, 0.203191 ,0.257844, 0.340244]
            intercept = 1.226555
            price = intercept + self.starfruit_mid_price_log[item][-4] * coef[0] + self.starfruit_mid_price_log[item][-3] * coef[1] + self.starfruit_mid_price_log[item][-2] * coef[2] + self.starfruit_mid_price_log[item][-1] * coef[3]
            return price
        else:
             return None
        
    def moving_average_list(self, item, period):
        if len(item) >= period:
            prices = item[-period:]
            sum_prices = sum(prices)
            return sum_prices / period
        
    def calculate_weighted_price(self, order_depth):
        weighted_bid = 0
        total_volume = 0
        weighted_ask = 0
        for price,volume in list(order_depth.buy_orders.items()):
            weighted_bid += price * volume
            total_volume += volume
        for price, volume in list(order_depth.sell_orders.items()):
            weighted_ask += price * -volume
            total_volume -= volume
        return (weighted_bid + weighted_ask) / total_volume
    
    def mid_price(self, order_depth):
        best_bid = sorted(list(order_depth.sell_orders.keys()))[0]
        best_ask = sorted(list(order_depth.buy_orders.keys()))[0]

        return (best_ask + best_bid) / 2
    
    def no_positions_basket(self, positions):
        return positions['CHOCOLATE'] == positions['ROSES'] == positions['STRAWBERRIES'] == 0
    
    def calculate_basket(self, state):
        order_depth = state.order_depths
        chocolate = "CHOCOLATE"
        strawberries = "STRAWBERRIES"
        roses = "ROSES"
        gift_basket = "GIFT_BASKET"
        etf_premium = 380
        threshold = 52
        orders = {
            chocolate: [],
            strawberries: [],
            roses: [],
            gift_basket: []
        }

        prices = {
            chocolate: self.calculate_weighted_price(order_depth[chocolate]),
            strawberries: self.calculate_weighted_price(order_depth[strawberries]),
            roses: self.calculate_weighted_price(order_depth[roses]),
            gift_basket: self.calculate_weighted_price(order_depth[gift_basket])
        }

        zero = 0
        best_ask_chocolate, best_ask_amount_chocolate = list(order_depth[chocolate].sell_orders.items())[zero]
        best_ask_strawberries, best_ask_amount_strawberries = list(order_depth[strawberries].sell_orders.items())[zero]
        best_ask_roses, best_ask_amount_roses = list(order_depth[roses].sell_orders.items())[zero]
        best_ask_gift, best_ask_amount_gift = list(order_depth[gift_basket].sell_orders.items())[zero]
        best_bid_chocolate, best_bid_amount_chocolate = list(order_depth[chocolate].buy_orders.items())[zero]
        best_bid_strawberries, best_bid_amount_strawberries = list(order_depth[strawberries].buy_orders.items())[zero]
        best_bid_roses, best_bid_amount_roses = list(order_depth[roses].buy_orders.items())[zero]
        best_bid_gift, best_bid_amount_gift = list(order_depth[gift_basket].buy_orders.items())[zero]

        combined_price = 4*prices[chocolate] + 6*prices[strawberries] + prices[roses]
        price_diff = prices[gift_basket] - combined_price
        self.basket_price_log['COMBINED'].append(combined_price)
        self.basket_price_log['BASKET'].append(prices[gift_basket])
        self.basket_price_log['DELTA'].append(price_diff)
        self.positions[chocolate] = state.position.get(chocolate, 0)
        self.positions[strawberries] = state.position.get(strawberries, 0)
        self.positions[roses] = state.position.get(roses, 0)
        self.positions[gift_basket] = state.position.get(gift_basket, 0)
        period = 2
        
        if len(self.basket_price_log['BASKET']) >= period:
            delta_curr = self.moving_average_list(self.basket_price_log['DELTA'], period)
                # if the etf prices exceeds the price of the individual products + the difference, then we short the products and long the etf
            if delta_curr < etf_premium - threshold:
                # should be positive
                units = min(best_bid_amount_chocolate//4, best_bid_amount_strawberries//6, best_bid_amount_roses, -best_ask_amount_gift)
                # long etf
                orders[gift_basket].append(Order(gift_basket, best_ask_gift, units))
            elif delta_curr > etf_premium + threshold:
                # positive
                units = min(-best_ask_amount_chocolate//4, -best_ask_amount_strawberries//6, -best_ask_amount_roses, best_bid_amount_gift)
                # short etf
                orders[gift_basket].append(Order(gift_basket, best_bid_gift, -units))

        else:
            if self.basket_price_log['DELTA'][-1] < etf_premium - threshold:
                # positive
                units = min(best_bid_amount_chocolate//4, best_bid_amount_strawberries//6, best_bid_amount_roses, -best_ask_amount_gift)
                # long etf
                orders[gift_basket].append(Order(gift_basket, best_ask_gift, units))
            elif self.basket_price_log['DELTA'][-1] > etf_premium + threshold:
                # positive
                units = min(-best_ask_amount_chocolate//4, -best_ask_amount_strawberries//6, -best_ask_amount_roses, best_bid_amount_gift)
                # short etf
                orders[gift_basket].append(Order(gift_basket, best_bid_gift, -units))

        return orders
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        for product in state.order_depths:
            
            if product == 'ORCHIDS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                self.positions[product] = state.position.get(product,0)

                buy_orders = list(order_depth.buy_orders.items())
                buy_orders.sort(key = lambda x:x[0], reverse = True)
                sell_orders = list(order_depth.sell_orders.items())
                sell_orders.sort(key = lambda x: x[0])
                best_bid, best_bid_amount = buy_orders[0]
                best_ask, best_ask_amount = sell_orders[0]
                storage_fee = 0.1
                #arbitrage
                orchids_observation = state.observations.conversionObservations['ORCHIDS']
                import_price = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees
                export_price = orchids_observation.bidPrice + orchids_observation.exportTariff + orchids_observation.transportFees

                if best_bid + 2 < import_price:
                    acc_price = round(import_price + 1)
                else:
                    acc_price = best_bid + 2
                orders.append(Order(product,acc_price, -100))
                if state.timestamp > 0:
                    conversions = -self.positions['ORCHIDS']
                result[product] = orders          
                
            elif product == 'AMETHYSTS':  
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                self.positions[product] = state.position.get(product,0)
                
                acceptable_price = 10000
                buy_limit = self.position_limit[product] - self.positions[product]
                sell_limit = - self.position_limit[product] - self.positions[product] #negative value
                bought_amount = 0
                sold_amount = 0
                
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limit[product] - self.positions[product],-best_ask_amount)
                            #logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                            if buy_amount == -best_ask_amount:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            #logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                    
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limit[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                            #logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                            if sell_amount == best_bid_amount:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            #logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                
                left_to_sell = sold_amount - sell_limit #positive number
                left_to_buy = buy_limit - bought_amount 

                #logger.print("MAKING BUY", str(left_to_buy) + "x", best_bid+1)
                orders.append(Order(product,best_bid+1,left_to_buy))
                #logger.print("MAKING SELL", str(left_to_sell) + "x", best_ask-1)
                orders.append(Order(product,best_ask-1,-left_to_sell))
                    

                            
                result[product] = orders
                #logger.print('\n'+str(self.positions[product]))
                #logger.print(state.position.get(product,0))

            elif product == 'STARFRUIT':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                self.positions[product] = state.position.get(product,0)
                extras = 20
                acceptable_price = self.calculate_weighted_price(order_depth)
                sold_amount = 0
                bought_amount = 0
                sell_limit = -self.position_limit[product] - self.positions[product]
                buy_limit = self.position_limit[product] - self.positions[product]

                # Append weighted price to mid-price log to calculate MA
                #self.starfruit_mid_price_log['STARFRUIT'].append(acceptable_price)
                window = 4
                #if len(self.starfruit_mid_price_log['STARFRUIT']) > window:
                    #acceptable_price = self.moving_average('STARFRUIT', window)
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        #i = 0
                        #i = 0
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limit[product] - self.positions[product],-best_ask_amount)
                            #logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                            # i = i+1
                            # if i < len(list(order_depth.sell_orders.items())):
                            #     best_ask, best_ask_amount = list(order_depth.sell_orders.items())[i]
                            # else:
                            #     break
                            # i = i+1
                            # if i < len(list(order_depth.sell_orders.items())):
                            #     best_ask, best_ask_amount = list(order_depth.sell_orders.items())[i]
                            # else:
                            #     break
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limit[product]:
                        i = 0
                        i = 0
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                            #logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                if best_bid <= acceptable_price-1 and best_ask >= acceptable_price+1:    
                    left_to_sell = sold_amount - sell_limit #positive number
                    left_to_buy = buy_limit - bought_amount 

                    #logger.print("MAKING BUY", str(left_to_buy) + "x", best_bid+1)
                    orders.append(Order(product,best_bid+1,left_to_buy))
                    #logger.print("MAKING SELL", str(left_to_sell) + "x", best_ask-1)
                    orders.append(Order(product,best_ask-1,-left_to_sell))
       
                result[product] = orders

        # orders for gift basket
        basket_orders = self.calculate_basket(state)
        for product in basket_orders:
            result[product] = basket_orders[product]
                     
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData