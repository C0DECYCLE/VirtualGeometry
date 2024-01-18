/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int } from "../../types/utils.type.js";

export function log(...data: any[]): void {
    return logger("log", ...data);
}

export function warn(...data: any[]): void {
    return logger("warn", ...data);
}

const logger_maximum: int = 256;
let logger_count: int = 0;

function logger(type: "log" | "warn", ...data: any[]): void {
    logger_count++;
    if (logger_count === logger_maximum + 1) {
        return console.warn("Logger: Maximum count exeeded.");
    } else if (logger_count < logger_maximum + 1) {
        return console[type](...data);
    }
}
