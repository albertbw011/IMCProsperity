from typing import Any
from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder
import json
from json import JSONEncoder
import numpy as np
import pandas as pd


class Trader:
    positions = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    position_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20} 
    
    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            
            if product == 'AMETHYSTS':  
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                acceptable_price = 10000
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price and self.positions['AMETHYSTS'] < self.position_limit['AMETHYSTS']:
                        buy_amount = min(self.position_limit['AMETHYSTS'] - self.positions['AMETHYSTS'],-best_ask_amount)
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product,best_ask,buy_amount))
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price and self.positions['AMETHYSTS'] > -self.position_limit['AMETHYSTS']:
                        sell_amount = min(self.positions['AMETHYSTS'] + self.position_limit['AMETHYSTS'], best_bid_amount)
                        print("SELL", str(sell_amount) + "x", best_bid)
                        orders.append(Order(product,best_bid,-sell_amount))
                result[product] = orders
                        
                    

                
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData