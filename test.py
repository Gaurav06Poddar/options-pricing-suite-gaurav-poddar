import importlib
import option_pricing
print("option_pricing imported:", option_pricing)
from option_pricing import BlackScholesModel, MonteCarloPricing
print("BlackScholesModel ok:", BlackScholesModel)
b = BlackScholesModel(100,100,30,0.01,0.2)
print("BS call price:", b.calculate_option_price("call"))
mc = MonteCarloPricing(100,100,30,0.01,0.2,number_of_simulations=1000)
mc.simulate_prices(seed=123)
print("MC price:", mc.calculate_option_price("call"))
