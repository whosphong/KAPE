from __future__ import annotations
import numpy as np


def createVDF(
    cdfData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
  qFlagKey = np.array([
      "Bit0: Counter Overflow",
      "Bit1: Survey Snapshot ON",
      "Bit2: Alternate Energy Table",
      "Bit3: Spoiler Test",
      "Bit4: Attenuator Engaged",
      "Bit5: Highest Archive Rate",
      "Bit6: No Targeted Sweep",
      "Bit7: SPAN-Ion New Mass Table",
      "Bit8: Over-deflection",
      "Bit9: Archive Snapshot ON",
      "Bit10: Bad energy table",
      "Bit11: MCP Test",
      "Bit12: Survey available",
      "Bit13: Archive available",
      "Bit14: RESERVED",
      "Bit15: RESERVED",
  ])

  mass_p = 0.01043970  # proton mass [eV/(km/s)^2]
  numEntries = len(cdfData["TIME"])

  eFlux = cdfData["EFlux"].reshape((numEntries, 8, 32, 8))
  counts = cdfData["DATA"].reshape((numEntries, 8, 32, 8))
  time_accum = cdfData["TIME_ACCUM"]

  gFactor = (
      counts / time_accum[:, np.newaxis, np.newaxis, np.newaxis]
  ) / eFlux

  energy = cdfData["ENERGY"].reshape((numEntries, 8, 32, 8))
  theta = cdfData["THETA"].reshape((numEntries, 8, 32, 8))
  phi = cdfData["PHI"].reshape((numEntries, 8, 32, 8))

  fillVals = np.nanmean(gFactor, axis=(1, 2, 3))
  for day_i in np.arange(0, np.shape(gFactor)[0]):
    gFactor[day_i, :, :, :] = np.nan_to_num(
        gFactor[day_i, :, :, :], nan=fillVals[day_i], copy=False
    )

  eFlux_one_count = (
      1 / time_accum[:, np.newaxis, np.newaxis, np.newaxis]
  ) / gFactor

  qualityFlags = np.unpackbits(
      cdfData["QUALITY_FLAG"].view(np.uint8), bitorder="little"
  )
  qFlags = qualityFlags.reshape(len(cdfData["QUALITY_FLAG"]), 16).astype(
      np.int32
  )

  reshaped_qFlags = np.tile(qFlagKey, len(cdfData["TIME"])).reshape(
      len(cdfData["TIME"]), 16
  )
  reshaped_qFlags[qFlags == 0] = np.nan

  numberFlux = eFlux / energy
  numberFlux_one_count = eFlux_one_count / energy

  vdf_temp = numberFlux * (mass_p**2) / ((2 * 10**5) * energy)
  vdf = vdf_temp * (100**3)

  vdf_temp_one_count = (
      numberFlux_one_count * (mass_p**2) / ((2 * 10**5) * energy)
  )
  vdf_one_count = vdf_temp_one_count * (100**3)

  vel = np.sqrt(2 * energy / mass_p)
  vx = vel * np.cos(np.radians(phi)) * np.cos(np.radians(theta))
  vy = vel * np.sin(np.radians(phi)) * np.cos(np.radians(theta))
  vz = vel * np.sin(np.radians(theta))

  return (reshaped_qFlags, vdf, vdf_one_count, counts, vx, vy, vz)


def calcRMatrix(L3Data) -> np.ndarray:
  B = L3Data["MAGF_INST"]
  B_mag = np.linalg.norm(B, axis=1, keepdims=True)
  j_prime = B / B_mag
  indexMin = j_prime == np.min(j_prime, 1, keepdims=True)
  j = np.zeros_like(j_prime)
  j[indexMin] = 1
  i_prime = np.cross(j_prime, j)
  i_prime /= np.linalg.norm(i_prime, axis=1, keepdims=True)
  k_prime = np.cross(i_prime, j_prime)
  k_prime /= np.linalg.norm(k_prime, axis=1, keepdims=True)
  return np.stack([i_prime, j_prime, k_prime], 1)