from enum import Enum
class ADF11Dataset(Enum): 
    EffectiveRecombinationCoefficients = "ACD"
    EffectiveIonisationCoefficients    = "SCD"
    CXCrossCouplingCoefficients        = "CCD"
    RecombinationAndBremsstrahlung     = "PRB"
    ChargeExchangeEmission             = "PRC"
    ParentCrossCouplingCoefficients    = "QCD"
    CrossCouplingCoefficients          = "XCD"
    LineEmissionFromExcitation         = "PLT"
    SpecificLineEmission               = "PLS"
    MeanChargeState                    = "ZCD"
    MeanChargeStateSquared             = "YCD"
    MeanIonisationPotential            = "ECD"

class ADF11Codes(Enum):
    EffectiveRecombinationCoefficients = 1
    EffectiveIonisationCoefficients    = 2
    CXCrossCouplingCoefficients        = 3
    RecombinationAndBremsstrahlung     = 4
    ChargeExchangeEmission             = 5
    ParentCrossCouplingCoefficients    = 6
    CrossCouplingCoefficients          = 7
    LineEmissionFromExcitation         = 8
    SpecificLineEmission               = 9
    MeanChargeState                    = 10
    MeanChargeStateSquared             = 11
    MeanIonisationPotential            = 12

class ADF11RateCoeffStoredUnits(Enum):
    EffectiveRecombinationCoefficients = "cm**3 / s"
    EffectiveIonisationCoefficients    = "cm**3 / s"
    LineEmissionFromExcitation         = "W cm**3"
    RecombinationAndBremsstrahlung     = "W cm**3"
    CXCrossCouplingCoefficients        = "cm**3 / s"
    ChargeExchangeEmission             = "W cm**3"
    MeanIonisationPotential            = "eV"

class ADF11RateCoeffDesiredUnits(Enum): 
    EffectiveRecombinationCoefficients = "m**3 / s"
    EffectiveIonisationCoefficients    = "m**3 / s"
    LineEmissionFromExcitation         = "W m**3"
    RecombinationAndBremsstrahlung     = "W m**3"
    CXCrossCouplingCoefficients        = "m**3 / s"
    ChargeExchangeEmission             = "W m**3"
    MeanIonisationPotential            = "eV"
