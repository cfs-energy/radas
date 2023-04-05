from enum import Enum
class ADF11Dataset(Enum): 
    EffectiveRecombinationCoefficients = "ACD"
    EffectiveIonisationCoefficients    = "SCD"
    CrossCouplingCoefficients          = "XCD"
    ParentCrossCouplingCoefficients    = "QCD"
    CXCrossCouplingCoefficients        = "CCD"
    LineEmissionFromExcitation         = "PLT"
    RecombinationAndBremsstrahlung     = "PRB"
    ChargeExchange                     = "PRC"
    SpecificLine                       = "PLS"

class ADF11Codes(Enum):
    EffectiveRecombinationCoefficients = 1
    EffectiveIonisationCoefficients    = 2
    CXCrossCouplingCoefficients        = 3
    RecombinationAndBremsstrahlung     = 4
    ChargeExchange                     = 5
    LineEmissionFromExcitation         = 8

class ADF11RateCoeffStoredUnits(Enum): 
    # Not correct yet!
    EffectiveRecombinationCoefficients = "cm**3/s"
    EffectiveIonisationCoefficients    = "cm**3/s"
    CXCrossCouplingCoefficients        = "cm**3/s"
    RecombinationAndBremsstrahlung     = "cm**3/s"
    ChargeExchange                     = "cm**3/s"
    LineEmissionFromExcitation         = "cm**3/s"

class ADF11RateCoeffDesiredUnits(Enum): 
    # Not correct yet!
    EffectiveRecombinationCoefficients = "m**3/s"
    EffectiveIonisationCoefficients    = "m**3/s"
    CXCrossCouplingCoefficients        = "m**3/s"
    RecombinationAndBremsstrahlung     = "m**3/s"
    ChargeExchange                     = "m**3/s"
    LineEmissionFromExcitation         = "m**3/s"