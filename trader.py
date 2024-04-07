from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
import numpy as np

class Trader:
    POSITION_LIMIT = {'BANANAS': 20, 'PEARLS': 20}

    def __init__(self):
        self.position = {product: 0 for product in self.POSITION_LIMIT}
        self.bananas_cache = []
        self.bananas_dim = 4

    def values_extract(self, order_dict, buy=0):
        total_volume = 0
        best_price = -1
        max_volume = -1

        for price, volume in order_dict.items():
            adjusted_volume = volume if buy else -volume
            total_volume += adjusted_volume
            if total_volume > max_volume:
                max_volume = total_volume
                best_price = price

        return total_volume, best_price

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        orders = []
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cpos = self.position[product]
        limit = self.POSITION_LIMIT[product]

        for price, volume in osell.items():
            if price <= acc_bid and cpos < limit:
                order_volume = min(-volume, limit - cpos)
                cpos += order_volume
                orders.append(Order(product, price, order_volume))

        for price, volume in obuy.items():
            if price >= acc_ask and cpos > -limit:
                order_volume = max(-volume, -limit - cpos)
                cpos += order_volume
                orders.append(Order(product, price, order_volume))

        self.position[product] = cpos  # Update position after orders
        return orders

    def calc_next_price_bananas(self):
        coef = [-0.01869561, 0.0455032, 0.16316049, 0.8090892]
        intercept = 4.481696494462085
        next_price = intercept
        for i, val in enumerate(self.bananas_cache):
            next_price += val * coef[i]
        return int(round(next_price))

    def run(self, state: TradingState):
        result = {product: [] for product in self.POSITION_LIMIT}

        for product in self.POSITION_LIMIT:
            self.position[product] = state.position.get(product, 0)

        if len(self.bananas_cache) == self.bananas_dim:
            self.bananas_cache.pop(0)
        sell_prices, buy_prices = [], []

        for product in ['BANANAS', 'PEARLS']:
            order_depth = state.order_depths[product]
            _, best_sell = self.values_extract(collections.OrderedDict(sorted(order_depth.sell_orders.items())))
            _, best_buy = self.values_extract(collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True)), 1)
            sell_prices.append(best_sell)
            buy_prices.append(best_buy)

        self.bananas_cache.append((sell_prices[0] + buy_prices[0]) / 2)

        acc_bid = {product: 10 for product in self.POSITION_LIMIT}  # Placeholder for actual logic
        acc_ask = {product: 12 for product in self.POSITION_LIMIT}  # Placeholder for actual logic

        if len(self.bananas_cache) == self.bananas_dim:
            bananas_price = self.calc_next_price_bananas()
            acc_bid['BANANAS'] = bananas_price - 1
            acc_ask['BANANAS'] = bananas_price + 1

        for product in ['BANANAS', 'PEARLS']:
            orders = self.compute_orders(product, state.order_depths[product], acc_bid[product], acc_ask[product])
            result[product] += orders

        return result

# Note: Ensure all required objects and methods from datamodel are correctly implemented to work with this trader logic.
