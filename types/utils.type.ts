/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

export type int = number & { __type?: "int" };

export type float = number & { __type?: "float" };

export type uuid = string & { __type?: "uuid" };

export type Nullable<T> = T | null;

export type Undefinable<T> = T | undefined;

export type EmptyCallback = () => void;

export type FloatArray = float[] | Float32Array | Float64Array;

export type Modify<T, R> = Omit<T, keyof R> & R;

export class MapString<T> extends Map<string, T> {}
