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

empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE' : 0, 'STRAWBERRIES': 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}


def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE' : 0, 'STRAWBERRIES': 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}
    position_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE' : 250, 'STRAWBERRIES': 350, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    mid_price_log = {'STARFRUIT': [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
    price_log = {'STARFRUIT':[], 'ORCHIDS' : []}
    orchids_stats = {'SUNLIGHT' : [], 'HUMIDITY': [], 'EXPORT TARIFFS': [], 'IMPORT TARIFFS': [], 'SHIPPING COSTS': []}
    trending = {'SUNLIGHT' : 0, 'HUMIDITY' : 0}


    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE' : 250, 'STRAWBERRIES': 350, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = collections.defaultdict(def_value)
    person_actvalof_position = collections.defaultdict(def_value)

    cpnl = collections.defaultdict(lambda : 0)
    bananas_cache = []
    coconuts_cache = []
    bananas_dim = 4
    coconuts_dim = 3
    steps = 0
    last_dolphins = -1
    buy_gear = False
    sell_gear = False
    buy_berries = False
    sell_berries = False
    close_berries = False
    last_dg_price = 0
    start_berries = 0
    first_berries = 0
    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    
    halflife_diff = 5
    alpha_diff = 1 - np.exp(-np.log(2)/halflife_diff)

    halflife_price = 5
    alpha_price = 1 - np.exp(-np.log(2)/halflife_price)

    halflife_price_dip = 20
    alpha_price_dip = 1 - np.exp(-np.log(2)/halflife_price_dip)
    
    begin_diff_dip = -INF
    begin_diff_bag = -INF
    begin_bag_price = -INF
    begin_dip_price = -INF

    std = 25
    basket_std = 103
    
    def moving_average(self, item, period):
        '''
        Calculate the weighted moving average price for the specified item over the specified period.
        '''

        # Check if the item exists in the mid_price_log dictionary 
        # and has enough entries to calculate the moving average for the given period.
        if item in self.mid_price_log and len(self.mid_price_log[item]) >= period:

            # Extract the last 'period' number of prices for the specified item.
            prices = self.mid_price_log[item][-period:]
            
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
            price = intercept + self.mid_price_log[item][-4] * coef[0] + self.mid_price_log[item][-3] * coef[1] + self.mid_price_log[item][-2] * coef[2] + self.mid_price_log[item][-1] * coef[3]
            return price
        else:
            
            
            return None
        
    # def calculate_slope(self, datapts, period):
        
    #     # Ensure that the period is valid
    #     if period <= 0 or period >= len(datapts):
    #         return None
        
    #     # Get the most recent data point
    #     y2 = datapts[-1]
        
    #     # Get the data point from the given period ago
    #     y1 = datapts[-period - 1]
        
    #     # Calculate the slope using the two data points
    #     slope = (y2 - y1) / period
        
    #     return slope
        
    
    # def set_trending(self, datapts):
    #     small_slope = self.calculate_slope(datapts, 3)
    #     medium_slope = self.calculate_slope(datapts, 5)
    #     large_slope = self.calculate_slope(datapts, 10)

    #     if small_slope > medium_slope and medium_slope > large_slope:
    #         self.trending = 1
    #     elif small_slope < medium_slope and medium_slope < large_slope:
    #         self.trending = -1
    #     else:
    #         self.trending = 0
    def compute_orders_basket2(self, order_depth):

        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            # vol_buy[p], vol_sell[p] = 0, 0
            # for price, vol in obuy[p].items():
            #     vol_buy[p] += vol 
            #     if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
            #         break
            # for price, vol in osell[p].items():
            #     vol_sell[p] += -vol 
            #     if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
            #         break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 375
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 375

        trade_at = self.basket_std*0.5
        # close_at = self.basket_std*(-1000)

        # pb_pos = self.position['GIFT_BASKET']
        # pb_neg = self.position['GIFT_BASKET']

        # uku_pos = self.position['ROSES']
        # uku_neg = self.position['ROSES']


        # basket_buy_sig = 0
        # basket_sell_sig = 0

        # if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
        #     self.cont_buy_basket_unfill = 0
        # if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
        #     self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            # self.cont_buy_basket_unfill = 0 # no need to buy rn
            # assert(vol >= 0)
            if vol > 0:
                # do_bask = 1
                # basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -int(vol/15))) 
                # self.cont_sell_basket_unfill += 2
                #pb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            # self.cont_sell_basket_unfill = 0 # no need to sell rn
            # assert(vol >= 0)
            if vol > 0:
                # do_bask = 1
                # basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], int(vol/15)))
                # self.cont_buy_basket_unfill += 2
                #pb_pos += vol

        return orders

    def determine_trend(self, value, datapts, period):
        """
        Determines whether a stock is increasing, decreasing, or ranging over a given period,
        based on all points within the period.
        
        Args:
            datapts (list): A list of data points (e.g., stock prices).
            period (int): The number of data points to consider for determining the trend.
        
        Returns:
            str: "Increasing" if the stock is increasing over the given period,
                "Decreasing" if the stock is decreasing over the given period,
                "Ranging" if the stock is ranging (neither increasing nor decreasing) over the given period.
        """
        # Ensure that the period is valid
        if period <= 1 or period > len(datapts):
            return
        
        # Get the data points for the given period
        period_data = datapts[-period:]
        
        # Calculate the price changes between consecutive points
        price_changes = [period_data[i] - period_data[i - 1] for i in range(1, period)]
        
        # Count the number of positive, negative, and zero price changes
        positive_changes = sum(1 for change in price_changes if change > 0)
        negative_changes = sum(1 for change in price_changes if change < 0)
        zero_changes = period - positive_changes - negative_changes
        
        # Determine the trend based on the counts
        if positive_changes > negative_changes and positive_changes > zero_changes:
            self.trending[value] = 1
        elif negative_changes > positive_changes and negative_changes > zero_changes:
            self.trending[value] = -1
        else:
            self.trending[value] = 0
        


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
    
    def compute_orders_basket(self, order_depths):
        products = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        mid_price, best_ask, best_ask_amount, best_bid, best_bid_amount, buy_amount, sell_amount  = {}, {}, {}, {}, {}, {}, {}
        coeffs = {'CHOCOLATE' : 0.1142, 'STRAWBERRIES': 0.0583, 'ROSES' : 0.207, 'GIFT_BASKET' : 1}

        for product in products:
            order_depth: OrderDepth = order_depths[product]
            mid_price[product] = self.calculate_weighted_price(order_depth)
            self.mid_price_log[product].append(self.calculate_weighted_price(order_depth))
            if len(order_depth.sell_orders) != 0:
                best_ask[product], best_ask_amount[product] = list(order_depth.sell_orders.items())[0]
                buy_amount[product] = min(self.position_limit[product] - self.positions[product],-best_ask_amount[product])
            if len(order_depth.buy_orders) != 0:
                best_bid[product], best_bid_amount[product] = list(order_depth.buy_orders.items())[0]
                sell_amount[product] = min(self.positions[product] + self.position_limit[product], best_bid_amount[product])
                                    
        
        diff = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES']
        print('gift basket')
        print(mid_price['GIFT_BASKET'])
        print('choc')
        print(mid_price['CHOCOLATE'])
        print('ROSES')
        print(mid_price['ROSES'])
        print('STRAWBERRIES')
        print(mid_price['STRAWBERRIES'])

        #hoc_coeff = 0.113
        #print('bert2')


        for product in products:
            print('bert3')
            if mid_price[product] < (mid_price['GIFT_BASKET']-375)*coeffs[product]*0.98:
                orders[product].append(Order(product, best_ask[product], buy_amount[product]))
                self.positions[product] += buy_amount[product]
            elif mid_price[product] > (mid_price['GIFT_BASKET']-375)*coeffs[product]*1.02:
                orders[product].append(Order(product, best_bid[product], -sell_amount[product]))
                self.positions[product] += -sell_amount[product]
                

        return orders
    
    

    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        time = state.timestamp
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
                self.mid_price_log['STARFRUIT'].append(acceptable_price)
                window = 4
                if len(self.mid_price_log['STARFRUIT']) > window:
                    acceptable_price = self.moving_average('STARFRUIT', window)
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limit[product]:
                        #i = 0
                        #i = 0
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limit[product] - self.positions[product],-best_ask_amount)
                            logger.print("BUY", str(buy_amount) + "x", best_ask)
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
                #if best_ask - best_bid < 7:
                    #orders.append(Order(product,best_bid+1, -100))
                #else:
                if best_bid+2 > import_price:
                    orders.append(Order(product,best_bid+2, -100))
                    if state.timestamp > 0:
                        conversions = -self.positions['ORCHIDS']
                result[product] = orders 
                # orders: list[Order] = []
                # order_depth = state.order_depths['ORCHIDS']
                # self.positions['ORCHIDS'] = state.position.get(product,0)
                # buy_orders = list(order_depth.buy_orders.items())
                # buy_orders.sort(key = lambda x:x[0], reverse = True)
                # sell_orders = list(order_depth.sell_orders.items())
                # sell_orders.sort(key = lambda x: x[0])
                # best_bid, best_bid_amount = buy_orders[0]
                # best_ask, best_ask_amount = sell_orders[0]
                # second_bid, second_bid_amount = buy_orders[1]
                # storage_fee = 0.1

                # self.orchids_stats['IMPORT TARIFFS'].append(state.observations.conversionObservations['ORCHIDS'].importTariff)
                # self.orchids_stats['EXPORT TARIFFS'].append(state.observations.conversionObservations['ORCHIDS'].exportTariff)
                # self.orchids_stats['SHIPPING COSTS'].append(state.observations.conversionObservations['ORCHIDS'].transportFees)
                # self.orchids_stats['SUNLIGHT'].append(state.observations.conversionObservations['ORCHIDS'].sunlight)
                # self.orchids_stats['HUMIDITY'].append(state.observations.conversionObservations['ORCHIDS'].sunlight)

                # #arbitrage
                # orchids_observation = state.observations.conversionObservations['ORCHIDS']
                # import_price = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees
                # export_price = orchids_observation.bidPrice + orchids_observation.exportTariff + orchids_observation.transportFees
                # logger.print(import_price)


                # #selling at best bid
                # #want to buy back for lower

                # spread = best_ask - best_bid
                # high_volume = spread
                # if high_volume > 100:
                #     high_volume = 100
                
                
                # one_volume = 10-spread
                # if one_volume < 0:
                #     one_volume =  0

                # low_volume  = 100 - high_volume - one_volume
                
                # orders.append(Order(product,best_bid+3, -high_volume))
                # orders.append(Order(product,best_bid+2, -low_volume))
                # orders.append(Order(product,best_bid+1, -one_volume))

                # if state.timestamp > 0:
                #     conversions = -self.positions['ORCHIDS']
                # result[product] = orders


                # order_depth: OrderDepth = state.order_depths[product]
                # orders: list[Order] = []
                # self.positions[product] = state.position.get(product,0)
                # acceptable_price = self.calculate_weighted_price(order_depth)

                # self.mid_price_log['ORCHIDS'].append(acceptable_price)

                # # self.determine_trend('HUMIDITY', self.orchids_stats['HUMIDITY'], 10)
                # # self.determine_trend('SUNLIGHT', self.orchids_stats['SUNLIGHT'], 10)

                # result[product] = orders
        
        orders = self.compute_orders_basket2(state.order_depths)
        if len(orders['GIFT_BASKET']) != 0:
            print('bert3')
            result['GIFT_BASKET'] = orders['GIFT_BASKET']
        result['CHOCOLATE'] = orders['CHOCOLATE']
        result['STRAWBERRIES'] = orders['STRAWBERRIES']
        result['ROSES'] = orders['ROSES']

                     
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
