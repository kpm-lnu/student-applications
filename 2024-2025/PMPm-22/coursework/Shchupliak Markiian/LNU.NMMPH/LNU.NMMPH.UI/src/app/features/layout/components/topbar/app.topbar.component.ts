import { Component, OnInit, ViewChild } from '@angular/core';
import { MenuItem } from 'primeng/api';
import { Menu } from 'primeng/menu';
import { MenuItems } from './static/menu-items';
import { LayoutService } from '../../services/app.layout.service';

@Component({
  selector: 'app-topbar',
  templateUrl: './app.topbar.component.html'
})
export class AppTopBarComponent {
  desktopMenu: any[] = [{ items: MenuItems.slice() }];
  mobileMenu: MenuItem[] = MenuItems.slice();

  @ViewChild('menu') menu: Menu | null = null;

  constructor(public layoutService: LayoutService) { }

  toggleMenu(event: MouseEvent) {
    this.menu!.toggle(event);
  }
}
