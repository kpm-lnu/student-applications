import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AppLayoutComponent } from './features/layout/components/layout/app.layout.component';

const routes: Routes = [{
  path: '',
  component: AppLayoutComponent,
  children: [
    {
      path: '',
      loadChildren: () => import('./features/methods/methods.module').then(m => m.MethodsModule)
    }]
}];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
