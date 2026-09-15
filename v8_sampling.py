"""Deterministic station-balanced V8 batch schedule."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BalancedBatch:
    epoch: int
    station_round: int
    batch_in_round: int
    batch_index: int
    stations: np.ndarray       # [4]
    times: np.ndarray          # [4,64]
    station_indices: np.ndarray  # flattened [256]
    time_indices: np.ndarray     # flattened [256]


class BalancedStationTimeSchedule:
    """Four stations x 64 valid times, 128 full-station rounds per epoch.

    Every station round independently permutes all stations and chunks that
    permutation into groups of four.  Thus every station occurs exactly once
    per round and exactly 128 times per epoch.  Every station owns an
    independent shuffled valid-time cycle;
    no time repeats until that station's current cycle is exhausted.
    ``state_dict`` captures all RNG/order/cursor state needed for exact resume.
    """

    stations_per_batch = 4
    times_per_station = 64
    rounds_per_epoch = 128

    def __init__(
        self,
        valid_times: Mapping[int, Sequence[int] | np.ndarray],
        *,
        seed: int = 42,
    ) -> None:
        if len(valid_times) < self.stations_per_batch:
            raise ValueError("at least four stations are required")
        self.stations = np.asarray(sorted(int(s) for s in valid_times), dtype=np.int64)
        if len(np.unique(self.stations)) != len(self.stations):
            raise ValueError("station identifiers must be unique")
        if len(self.stations) % self.stations_per_batch:
            raise ValueError("station count must be divisible by four")
        self._valid_times: dict[int, np.ndarray] = {}
        for station in self.stations:
            values = np.asarray(valid_times[int(station)], dtype=np.int64).reshape(-1)
            if values.size == 0:
                raise ValueError(f"station {station} has no valid target timestamps")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"station {station} valid timestamps contain duplicates")
            self._valid_times[int(station)] = values.copy()

        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.epoch = 0
        self.station_round = 0
        self.batch_in_round = 0
        self._station_order = self._rng.permutation(self.stations)
        self._time_orders = {
            int(station): self._rng.permutation(self._valid_times[int(station)])
            for station in self.stations
        }
        self._time_cursors = {int(station): 0 for station in self.stations}

    def __len__(self) -> int:
        return self.rounds_per_epoch * self.batches_per_station_round

    @property
    def batches_per_station_round(self) -> int:
        return len(self.stations) // self.stations_per_batch

    @property
    def samples_per_batch(self) -> int:
        return self.stations_per_batch * self.times_per_station

    def _take_times(self, station: int, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            order = self._time_orders[station]
            cursor = self._time_cursors[station]
            available = len(order) - cursor
            take = min(remaining, available)
            if take:
                pieces.append(order[cursor : cursor + take])
                cursor += take
                remaining -= take
            if cursor == len(order):
                order = self._rng.permutation(self._valid_times[station])
                cursor = 0
            self._time_orders[station] = order
            self._time_cursors[station] = cursor
        return np.concatenate(pieces).astype(np.int64, copy=False)

    def _finish_epoch(self) -> None:
        self.epoch += 1
        self.station_round = 0
        self.batch_in_round = 0
        self._station_order = self._rng.permutation(self.stations)

    def next_batch(self) -> BalancedBatch:
        epoch = self.epoch
        station_round = self.station_round
        batch_in_round = self.batch_in_round
        batch_index = station_round * self.batches_per_station_round + batch_in_round
        start = batch_in_round * self.stations_per_batch
        chosen = self._station_order[start : start + self.stations_per_batch].astype(
            np.int64, copy=True
        )
        if len(np.unique(chosen)) != self.stations_per_batch:
            raise AssertionError("balanced station cycle produced a duplicate station")
        times = np.stack(
            [self._take_times(int(station), self.times_per_station) for station in chosen],
            axis=0,
        )
        result = BalancedBatch(
            epoch=epoch,
            station_round=station_round,
            batch_in_round=batch_in_round,
            batch_index=batch_index,
            stations=chosen,
            times=times,
            station_indices=np.repeat(chosen, self.times_per_station),
            time_indices=times.reshape(-1),
        )
        self.batch_in_round += 1
        if self.batch_in_round == self.batches_per_station_round:
            self.batch_in_round = 0
            self.station_round += 1
            if self.station_round == self.rounds_per_epoch:
                self._finish_epoch()
            else:
                # Each full station round receives an independent permutation.
                self._station_order = self._rng.permutation(self.stations)
        return result

    def iter_epoch(self):
        """Yield the remaining rounds of the current epoch."""
        epoch = self.epoch
        while self.epoch == epoch:
            yield self.next_batch()

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "seed": self.seed,
            "stations": self.stations.copy(),
            "valid_times": {k: v.copy() for k, v in self._valid_times.items()},
            "epoch": self.epoch,
            "station_round": self.station_round,
            "batch_in_round": self.batch_in_round,
            "station_order": self._station_order.copy(),
            "time_orders": {k: v.copy() for k, v in self._time_orders.items()},
            "time_cursors": dict(self._time_cursors),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported sampler state version")
        if not np.array_equal(np.asarray(state["stations"], dtype=np.int64), self.stations):
            raise ValueError("sampler state stations do not match")
        for station in self.stations:
            key = int(station)
            if not np.array_equal(
                np.asarray(state["valid_times"][key], dtype=np.int64), self._valid_times[key]
            ):
                raise ValueError(f"sampler valid timestamps changed for station {key}")
        epoch = int(state["epoch"])
        station_round = int(state["station_round"])
        batch_in_round = int(state["batch_in_round"])
        if epoch < 0 or not 0 <= station_round < self.rounds_per_epoch:
            raise ValueError("invalid epoch/station round in sampler state")
        if not 0 <= batch_in_round < self.batches_per_station_round:
            raise ValueError("invalid batch position in sampler state")
        station_order = np.asarray(state["station_order"], dtype=np.int64)
        if sorted(station_order.tolist()) != sorted(self.stations.tolist()):
            raise ValueError("invalid station order in sampler state")

        restored_orders: dict[int, np.ndarray] = {}
        restored_cursors: dict[int, int] = {}
        for station in self.stations:
            key = int(station)
            order = np.asarray(state["time_orders"][key], dtype=np.int64)
            if sorted(order.tolist()) != sorted(self._valid_times[key].tolist()):
                raise ValueError(f"invalid time order for station {key}")
            cursor = int(state["time_cursors"][key])
            if not 0 <= cursor < len(order):
                raise ValueError(f"invalid time cursor for station {key}")
            restored_orders[key] = order.copy()
            restored_cursors[key] = cursor

        self.seed = int(state["seed"])
        self.epoch = epoch
        self.station_round = station_round
        self.batch_in_round = batch_in_round
        self._station_order = station_order.copy()
        self._time_orders = restored_orders
        self._time_cursors = restored_cursors
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = deepcopy(state["rng_state"])


V8BalancedSchedule = BalancedStationTimeSchedule
