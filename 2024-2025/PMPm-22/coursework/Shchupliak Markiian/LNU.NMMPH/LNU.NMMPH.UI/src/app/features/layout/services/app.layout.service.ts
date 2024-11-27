import { Injectable } from '@angular/core';

@Injectable({
    providedIn: 'root',
})
export class LayoutService {

    constructor() { }

    isDesktop() {
        return window.innerWidth > 1280;
    }

    isMobile() {
        return !this.isDesktop();
    }
}
