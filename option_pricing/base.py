# option_pricing/base.py
from enum import Enum
from abc import ABC, abstractmethod

class OPTION_TYPE(Enum):
    CALL_OPTION = 'call'
    PUT_OPTION = 'put'

class OptionPricingModel(ABC):
    """Abstract class defining interface for option pricing models."""

    def calculate_option_price(self, option_type):
        """
        Accepts many common option_type forms:
         - 'call', 'put' (case-insensitive)
         - 'Call Option', 'Put Option'
         - OPTION_TYPE enum values
        """
        if isinstance(option_type, OPTION_TYPE):
            opt = option_type.value
        elif isinstance(option_type, str):
            opt = option_type.strip().lower()
            if opt == 'call option':
                opt = 'call'
            elif opt == 'put option':
                opt = 'put'
        else:
            opt = str(option_type).lower()

        if opt.startswith('call'):
            return self._calculate_call_option_price()
        elif opt.startswith('put'):
            return self._calculate_put_option_price()
        else:
            raise ValueError(f"Unsupported option_type: {option_type}")

    @abstractmethod
    def _calculate_call_option_price(self):
        """Calculates option price for call option."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_put_option_price(self):
        """Calculates option price for put option."""
        raise NotImplementedError()
