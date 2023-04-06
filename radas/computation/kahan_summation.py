"""Summation algorithms which don't lose precision when differencing large numbers."""

def kahan_summation(values_to_sum):
    """Basic Kahan compensated summation algorithm."""

    running_sum = 0.0
    compensation = 0.0

    for value in values_to_sum:

        compensated_value = value - compensation
        uncompensated_sum = running_sum + compensated_value
        compensation = (uncompensated_sum - running_sum) - compensated_value
        running_sum = uncompensated_sum
    
    return running_sum

def kahan_babushka_neumaier_sum(values_to_sum):
    """Improved Kahan compensated summation algorithm."""
    running_sum = 0.0
    compensation = 0.0

    for value in values_to_sum:
        temporary_sum = running_sum + value

        if running_sum >= value:
            compensation += running_sum - temporary_sum + value
        else:
            compensation += value - temporary_sum + running_sum
        
        running_sum = temporary_sum

    return running_sum + compensation
