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
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0,'ORCHIDS': 0, 'GIFT_BASKET': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'COCONUTS': 0, 'COCONUT_COUPON': 0}
    position_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20,'ORCHIDS': 100, 'GIFT_BASKET': 60, 'CHOCOLATE': 250, 'STRAWBERRIES': 360, 'ROSES': 60, 'COCONUTS': 300, 'COCONUT_COUPON': 600}
    starfruit_mid_price_log = {'STARFRUIT': []}
    starfruit_price_log = {'STARFRUIT':[]}
    baskets_mp = {'GB' : [],'CHOC' :[], 'STRAW': [], 'ROSE': [], 'SUM': []}
    basket_price_log = {'COMBINED': [], 'BASKET': [], 'DELTA': []}
    
    def moving_average(self, item, period):
        '''
        Calculate the weighted moving average price for the specified item over the specified period.
        '''
        if item in self.starfruit_mid_price_log and len(self.starfruit_mid_price_log[item]) >= period:

            prices = self.starfruit_mid_price_log[item][-period:]
            
            weights = range(1, period + 1)
            
            weighted_sum = sum(price * weight for price, weight in zip(prices, weights))
            
            total_weight = sum(weights)
            
            coef = [0.346080, 0.262699 ,0.195654, 0.192134]
            intercept = 17.36839
            price = intercept + self.starfruit_mid_price_log[item][-4] * coef[0] + self.starfruit_mid_price_log[item][-3] * coef[1] + self.starfruit_mid_price_log[item][-2] * coef[2] + self.starfruit_mid_price_log[item][-1] * coef[3]
            return price
        else:
            return None
        
    def MA_baskets(self,period,product):
        total_sum = sum(self.baskets_mp[product][-period:])
        return total_sum/period
    
    def moving_average_list(self, item, period):
        if len(item) >= period:
            prices = item[-period:]
            sum_prices = sum(prices)
            return sum_prices / period    
        
    def compute_order_coconuts(self, order_depth, timestamp):
        orders = {'COCONUT': [], 'COCONUT_COUPON': []}
        coupon_asks = list(order_depth['COCONUT_COUPON'].sell_orders.items())
        coupon_bids = list(order_depth['COCONUT_COUPON'].buy_orders.items())
        coconut_asks = list(order_depth['COCONUT'].sell_orders.items())
        coconut_bids = list(order_depth['COCONUT'].buy_orders.items())
        
        stock_price = (coconut_asks[0][0] + coconut_bids[0][0])/2
        strike_price = 10000 #price the call option can be executed at
        time_to_expiration = (247 - timestamp/1000000)/365 #time till expiration in years
        risk_free_rate = 0 #no banks on the island
        sigma = 0.19332951334290546 #calculated using premium on day 1 timestamp 1, might need to check this
        
        fair_price = self.black_scholes_price(stock_price,strike_price,time_to_expiration, risk_free_rate,sigma)
        logger.print(fair_price)
        bought_amount = 0
        sold_amount = 0
        
        if len(coupon_asks) != 0:
            if fair_price > coupon_asks[0][0]:
                ask_price, ask_volume = coupon_asks[0]
                if self.positions['COCONUT_COUPON'] < self.position_limit['COCONUT_COUPON']:
                    buy_amount = min(self.position_limit['COCONUT_COUPON'] - self.positions['COCONUT_COUPON'],-ask_volume)
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON',ask_price, buy_amount))
                    self.positions['COCONUT_COUPON'] += buy_amount
                    bought_amount += buy_amount

        if len(coupon_bids) != 0:
            if fair_price < coupon_bids[0][0]:
                bid_price, bid_volume = coupon_bids[0]
                if self.positions['COCONUT_COUPON'] > -self.position_limit['COCONUT_COUPON']:
                    sell_amount = min(self.positions['COCONUT_COUPON'] + self.position_limit['COCONUT_COUPON'], bid_volume)
                    logger.print("SELL", str(sell_amount) + "x", bid_price)
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON',bid_price,-sell_amount))
                    self.positions['COCONUT_COUPON'] -= sell_amount
                    sold_amount -= sell_amount
        
        #can try market making, not sure if bots will take, spread is also very small, nearly 0
        delta = self.delta(stock_price,strike_price,time_to_expiration, risk_free_rate,sigma) * self.positions['COCONUT_COUPON'] + self.positions['COCONUT']
        value = 200
        if delta > value:
            bid_price, bid_volume = coconut_bids[0]
            sell_amount = min(int(delta - value), bid_volume)
            orders['COCONUT'].append(Order('COCONUT',bid_price, -sell_amount))
        if delta < -value:
            ask_price, ask_volume = coconut_asks[0]
            buy_amount = min(int(-value - delta),-ask_volume)
            orders['COCONUT'].append(Order('COCONUT',ask_price, buy_amount))
        
        
                
        return orders    
        
    def no_positions_basket(self, positions):
        return positions['CHOCOLATE'] == positions['ROSES'] == positions['STRAWBERRIES'] == 0
    
    def calculate_basket(self, state):
        order_depth = state.order_depths
        chocolate = "CHOCOLATE"
        strawberries = "STRAWBERRIES"
        roses = "ROSES"
        gift_basket = "GIFT_BASKET"
        etf_premium = 386
        threshold = 38
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
    
    def black_scholes_price(self,S,K,T,r,sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * self.phi(d1) - K * np.exp(-r * T) * self.phi(d2)
        return call_price
    
    def phi(self,x):
    #'Cumulative distribution function for the standard normal distribution'
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    def delta(self,S,K,T,r,sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return self.phi(d1)
    
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
    
        
        
    def get_mid_price(self,order_depth):
        buy_orders = list(order_depth.buy_orders.items())
        buy_orders.sort(key = lambda x:x[0], reverse = True)
        sell_orders = list(order_depth.sell_orders.items())
        sell_orders.sort(key = lambda x: x[0])
        best_bid, best_bid_amount = buy_orders[0]
        best_ask, best_ask_amount = sell_orders[0]
        return (best_bid + best_ask) /2
        
    

    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        for product in state.order_depths:
            
            if product == 'AMETHYSTS':  
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
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                            if buy_amount == -best_ask_amount:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                    
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limit[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                            logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                            if sell_amount == best_bid_amount:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                
                left_to_sell = sold_amount - sell_limit #positive number
                left_to_buy = buy_limit - bought_amount 

                logger.print("MAKING BUY", str(left_to_buy) + "x", best_bid+1)
                orders.append(Order(product,best_bid+1,left_to_buy))
                logger.print("MAKING SELL", str(left_to_sell) + "x", best_ask-1)
                orders.append(Order(product,best_ask-1,-left_to_sell))
                    

                            
                result[product] = orders
                logger.print('\n'+str(self.positions[product]))
                logger.print(state.position.get(product,0))
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
                self.starfruit_mid_price_log['STARFRUIT'].append(acceptable_price)
                window = 4
                if len(self.starfruit_mid_price_log['STARFRUIT']) > window:
                    acceptable_price = self.moving_average('STARFRUIT', window)
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limit[product] - self.positions[product],-best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limit[product]:
                        i = 0
                        i = 0
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limit[product], best_bid_amount)
                            logger.print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                if best_bid <= acceptable_price-1 and best_ask >= acceptable_price+1:    
                    left_to_sell = sold_amount - sell_limit #positive number
                    left_to_buy = buy_limit - bought_amount 

                    logger.print("MAKING BUY", str(left_to_buy) + "x", best_bid+1)
                    orders.append(Order(product,best_bid+1,left_to_buy))
                    logger.print("MAKING SELL", str(left_to_sell) + "x", best_ask-1)
                    orders.append(Order(product,best_ask-1,-left_to_sell))
       
                result[product] = orders
            elif product == 'ORCHIDS':
                orders: list[Order] = []
                order_depth = state.order_depths['ORCHIDS']
                self.positions['ORCHIDS'] = state.position.get(product,0)
                buy_orders = list(order_depth.buy_orders.items())
                buy_orders.sort(key = lambda x:x[0], reverse = True)
                sell_orders = list(order_depth.sell_orders.items())
                sell_orders.sort(key = lambda x: x[0])
                best_bid, best_bid_amount = buy_orders[0]
                best_ask, best_ask_amount = sell_orders[0]
                second_bid, second_bid_amount = buy_orders[1]
                storage_fee = 0.1
                #arbitrage
                orchids_observation = state.observations.conversionObservations['ORCHIDS']
                import_price = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees
                export_price = orchids_observation.bidPrice + orchids_observation.exportTariff + orchids_observation.transportFees
                logger.print(import_price)
                #selling at best bid
                #want to buy back for lower
                if best_bid + 2 < import_price:
                    acc_price = round(import_price + 1)
                else:
                    acc_price = best_bid + 2
                orders.append(Order(product,acc_price, -100))
                if state.timestamp > 0:
                    conversions = -self.positions['ORCHIDS']
                result[product] = orders
            elif product == 'COCONUT_COUPON': 
                orders: list[Order] = []
                #order_depth = state.order_depths[product]
                self.positions['COCONUT_COUPON'] = state.position.get('COCONUT_COUPON',0)
                orders = self.compute_order_coconuts(state.order_depths,state.timestamp)
                result[product] = orders['COCONUT_COUPON']
            elif product == 'COCONUT':
                orders: list[Order] = []
                self.positions['COCONUT'] = state.position.get('COCONUT',0)
                orders = self.compute_order_coconuts(state.order_depths,state.timestamp)
                result[product] = orders['COCONUT']
                   
                
                
        """basket_orders = self.calculate_basket(state)
        for product in basket_orders:
            result[product] = basket_orders[product]    """                
                     
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData