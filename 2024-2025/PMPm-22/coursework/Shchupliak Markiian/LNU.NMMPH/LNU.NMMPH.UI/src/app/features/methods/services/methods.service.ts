import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class MethodsService {
  url: string = 'api/execute-methods';

  constructor(private http: HttpClient) { }

  postFile(formData: FormData): Observable<number> {
    return this.http.post<number>(`${environment.apiHost}/${this.url}`, formData);
  }

  getTemplate(fileName: string): Observable<Blob> {
    return this.http.get(`${environment.apiHost}/api/files/${fileName}`, { responseType: 'blob' });
  }
}
