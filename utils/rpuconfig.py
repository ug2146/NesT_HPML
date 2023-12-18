# Single device configs.
from aihwkit.simulator.presets import ReRamESPreset, ReRamSBPreset, EcRamPreset, PCMPreset

# 2-device configs.
from aihwkit.simulator.presets import ReRamES2Preset, EcRamMO2Preset

# 4-device configs.
from aihwkit.simulator.presets import ReRamSB4Preset, EcRam4Preset

# Tiki-taka configs.
from aihwkit.simulator.presets import TikiTakaReRamESPreset, TikiTakaEcRamPreset

# MixedPrecision configs.
from aihwkit.simulator.presets import MixedPrecisionReRamESPreset, MixedPrecisionEcRamMOPreset, MixedPrecisionPCMPreset

from aihwkit.simulator.presets import StandardHWATrainingPreset, CapacitorPresetDevice, EcRamMOPresetDevice
from aihwkit.simulator.presets import GokmenVlasovPresetDevice, PCMPresetDevice, ReRamArrayHfO2PresetDevice

from aihwkit.simulator.configs import MappingParameter

def get_rpuconfig(preset_parameters):

    key = preset_parameters['preset']
    mapping = MappingParameter(weight_scaling_omega=preset_parameters['weight_scaling_omega'], learn_out_scaling=preset_parameters['learn_out_scaling'])


    if 'tikitaka' in key.lower() and 'reram' in key.lower(): return TikiTakaReRamESPreset(mapping=mapping)
    if 'tikitaka' in key.lower() and 'ecram' in key.lower(): return TikiTakaEcRamPreset(mapping=mapping)

    if 'mixed' in key.lower() and 'reram' in key.lower(): return MixedPrecisionReRamESPreset(mapping=mapping)
    if 'mixed' in key.lower() and 'ecram' in key.lower(): return MixedPrecisionEcRamMOPreset(mapping=mapping)
    if 'mixed' in key.lower() and 'pcm' in key.lower(): return MixedPrecisionPCMPreset(mapping=mapping)

    if 'capacitor' in key.lower() and 'device' in key.lower(): return CapacitorPresetDevice(mapping=mapping)
    if 'ecrammo' in key.lower() and 'device' in key.lower(): return EcRamMOPresetDevice(mapping=mapping)
    if 'gokmen' in key.lower() and 'device' in key.lower(): return GokmenVlasovPresetDevice(mapping=mapping)
    if 'pcm' in key.lower() and 'device' in key.lower(): return PCMPresetDevice(mapping=mapping)
    if 'reram' in key.lower() and 'hfo2' in key.lower() and 'device' in key.lower(): return ReRamArrayHfO2PresetDevice(mapping=mapping)

    if 'rerames2' in key.lower(): return ReRamES2Preset(mapping=mapping)
    if 'ecrammo2' in key.lower(): return EcRamMO2Preset(mapping=mapping)
    if 'reramsb4' in key.lower(): return ReRamSB4Preset(mapping=mapping)
    if 'ecram4' in key.lower(): return EcRam4Preset(mapping=mapping)
    
    if 'rerames' in key.lower(): return ReRamESPreset(mapping=mapping)
    if 'reramsb' in key.lower(): return ReRamSBPreset(mapping=mapping)
    if 'ecram' in key.lower(): return EcRamPreset(mapping=mapping)
    if 'pcm' in key.lower(): return PCMPreset(mapping=mapping)
    
    
    
    