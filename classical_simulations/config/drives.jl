using Bessels
abstract type AbstractDrive end

drive_cos(ϵ::Float64, ω::Float64, t::Float64, phase::Float64) = ϵ * cos(ω * t + phase)

drive_JA_1(ϵ::Float64, ω::Float64, t::Float64, phase::Float64) = 2*besselj(1,ϵ)*cos(ω * t + phase)

drive_pump_full_coupling(ϵ::Float64, ω::Float64, t::Float64, phase::Float64) = sin(ϵ * cos(ω * t + phase))

abstract type DriveTwoModes <: AbstractDrive end
abstract type DriveTwoModes_prepared <: AbstractDrive end

mutable struct DriveTwoModesBasics <: DriveTwoModes
    ϵ_d::Float64
    ω_d::Float64
    buffer_phase::Float64
    ϵ_0::Float64
    ω_p::Float64
    pump_phase::Float64
end

mutable struct DriveTwoModesBasics_prepared{F1,F2} <: DriveTwoModes_prepared
    drive::DriveTwoModesBasics
    drive_buffer::F1
    drive_pump::F2
end

mutable struct DriveTwoModesJA <: DriveTwoModes
    ϵ_d::Float64
    ω_d::Float64
    buffer_phase::Float64
    ϵ_0::Float64
    ω_p::Float64
    pump_phase::Float64
end

mutable struct DriveTwoModesJA_prepared{F1,F2} <: DriveTwoModes_prepared
    drive::DriveTwoModesJA
    drive_buffer::F1
    drive_pump::F2
end


mutable struct DriveOneMode <: AbstractDrive
    ϵ::Float64
    ω::Float64
    phase::Float64
end

mutable struct DriveOneMode_prepared{F1} <: AbstractDrive
    drive::DriveOneMode
    drive_func::F1
end

function prepare(drive::AbstractDrive)
    throw(MethodError(prepare, (drive,)))
end

function prepare(drive::DriveTwoModesBasics)
    drive_prepared = DriveTwoModesBasics_prepared(drive, drive_cos, drive_pump_full_coupling)
    return drive_prepared
end

function prepare(drive::DriveTwoModesJA)
    drive_prepared = DriveTwoModesJA_prepared(drive, drive_cos, drive_JA_1)
    return drive_prepared
end

function prepare(drive::DriveOneMode)
    drive_prepared = DriveOneMode_prepared(drive, drive_cos)
    return drive_prepared
end