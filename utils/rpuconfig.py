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

def get_rpuconfig(key):
    if 'tikitaka' in key.lower() and 'reram' in key.lower(): return TikiTakaReRamESPreset
    if 'tikitaka' in key.lower() and 'ecram' in key.lower(): return TikiTakaEcRamPreset

    if 'mixed' in key.lower() and 'reram' in key.lower(): return MixedPrecisionReRamESPreset
    if 'mixed' in key.lower() and 'ecram' in key.lower(): return MixedPrecisionEcRamMOPreset
    if 'mixed' in key.lower() and 'pcm' in key.lower(): return MixedPrecisionPCMPreset

    if 'capacitor' in key.lower() and 'device' in key.lower(): return CapacitorPresetDevice
    if 'ecrammo' in key.lower() and 'device' in key.lower(): return EcRamMOPresetDevice
    if 'gokmen' in key.lower() and 'device' in key.lower(): return GokmenVlasovPresetDevice
    if 'pcm' in key.lower() and 'device' in key.lower(): return PCMPresetDevice
    if 'reram' in key.lower() and 'hfo2' in key.lower() and 'device' in key.lower(): return ReRamArrayHfO2PresetDevice

    if 'rerames2' in key.lower(): return ReRamES2Preset
    if 'ecrammo2' in key.lower(): return EcRamMO2Preset
    if 'reramsb4' in key.lower(): return ReRamSB4Preset
    if 'ecram4' in key.lower(): return EcRam4Preset
    
    if 'rerames' in key.lower(): return ReRamESPreset
    if 'reramsb' in key.lower(): return ReRamSBPreset
    if 'ecram' in key.lower(): return EcRamPreset
    if 'pcm' in key.lower(): return PCMPreset
    
    
    
    