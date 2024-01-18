/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, uuid } from "../../types/utils.type.js";

export const PHI: float = (1 + 5 ** 0.5) / 2;

export const toAngle: float = 180 / Math.PI;
export const toRadian: float = Math.PI / 180;

export function normalizeRadian(value: float): float {
    value = value % (2 * Math.PI);
    if (value < 0) {
        value += 2 * Math.PI;
    }
    return value;
}

export function UUIDv4(): uuid {
    return `${1e7}-${1e3}-${4e3}-${8e3}-${1e11}`.replace(/[018]/g, (c: any) =>
        (
            c ^
            (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
        ).toString(16),
    );
}

export function between(
    value: int | float,
    lower: int | float,
    upper: int | float,
): boolean {
    return value > Math.min(lower, upper) && value < Math.max(lower, upper);
}

export function dotit(value: int | float): string {
    return Math.round(value)
        .toString()
        .replace(/(\d)(?=(\d{3})+(?!\d))/g, "$1'");
}

export function clamp(
    value: int | float,
    min: int | float,
    max: int | float,
): float {
    return Math.min(Math.max(value, min), max) as float;
}

export function firstLetterUppercase(value: string): string {
    return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

export function replaceAt(
    value: string,
    index: int,
    replacement: string,
): string {
    return `${value.substring(0, index)}${replacement}${value.substring(
        index + replacement.length,
    )}`;
}
export function toHexadecimal(value: string): Number {
    return Number(`0x${value.split("#")[1]}`);
}

export function count<T>(value: T[], target: T): int {
    return value.filter((x: T): boolean => x === target).length;
}

export function clear<T>(value: T[]): T[] {
    value.length = 0;
    return value;
}

export function assert(condition: any, msg?: string): asserts condition {
    if (!condition) {
        throw new Error(msg);
    }
}
