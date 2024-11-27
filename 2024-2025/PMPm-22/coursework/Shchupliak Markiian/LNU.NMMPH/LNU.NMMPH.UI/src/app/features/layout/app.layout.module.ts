import { NgModule } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { BrowserModule } from "@angular/platform-browser";
import { RouterModule } from "@angular/router";
import { AppLayoutComponent } from "./components/layout/app.layout.component";
import { AppMenuitemComponent } from "./components/menu/app.menu-item.component";
import { MenubarModule } from 'primeng/menubar';
import { ToastModule } from 'primeng/toast';
import { ButtonModule } from 'primeng/button';
import { MenuModule } from 'primeng/menu';
import { AppTopBarComponent } from './components/topbar/app.topbar.component';

@NgModule({
  declarations: [
    AppMenuitemComponent,
    AppTopBarComponent,
    AppLayoutComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule,
    MenubarModule,
    ToastModule,
    ButtonModule,
    MenuModule
  ],
  exports: [AppLayoutComponent]
})
export class AppLayoutModule { }
