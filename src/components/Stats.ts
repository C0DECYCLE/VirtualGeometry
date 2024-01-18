/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float, Undefinable, MapString } from "../utilities/utils.type.js";

export class Stats {
    private readonly list: MapString<float>;

    private readonly div: HTMLDivElement;
    private readonly p: HTMLParagraphElement;

    public constructor() {
        this.list = new MapString<float>();
        this.div = this.createDiv();
        this.p = this.createP(this.div);
    }

    private createDiv(): HTMLDivElement {
        const div: HTMLDivElement = document.createElement("div");
        div.style.position = "absolute";
        div.style.top = "0px";
        div.style.right = "0px";
        div.style.minWidth = "30vh";
        div.style.backgroundColor = "#000000";
        div.style.opacity = "0.75";
        return div;
    }

    private createP(div: HTMLDivElement): HTMLParagraphElement {
        const p: HTMLDivElement = document.createElement("p");
        p.style.margin = "2vh";
        p.style.color = "#FFFFFF";
        p.style.fontFamily = "system-ui";
        p.style.fontSize = "1.2vh";
        div.append(p);
        return p;
    }

    public show(): void {
        document.body.appendChild(this.div);
    }

    public get(key: string): Undefinable<float> {
        return this.list.get(key);
    }

    public set(key: string, value: float): void {
        this.list.set(key, value);
    }

    public add(key: string, value: float): void {
        const current: Undefinable<float> = this.get(key);
        if (current === undefined) {
            throw new Error(`Stats: Unknown key. (${key})`);
        }
        this.list.set(key, current + value);
    }

    public time(key: string, keyDiff?: string | float): void {
        const now: float = performance.now();
        if (keyDiff === undefined) {
            this.list.set(key, now);
            return;
        }
        const sub: Undefinable<float> =
            typeof keyDiff === "string" ? this.get(keyDiff) : keyDiff;
        if (sub === undefined) {
            throw new Error(`Stats: Unknown key. (${key})`);
        }
        this.list.set(key, now - sub);
    }

    public update(text: string): void {
        this.p.innerHTML = text;
    }

    public destroy(): void {
        if (document.body.contains(this.div)) {
            document.body.removeChild(this.div);
        }
        this.list.clear();
    }
}
