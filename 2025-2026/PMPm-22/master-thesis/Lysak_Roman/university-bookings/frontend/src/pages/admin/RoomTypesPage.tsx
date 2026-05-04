import React, { useEffect, useState } from 'react';
import {
  makeStyles,
  Spinner,
  Badge,
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogActions,
  DialogContent,
  Field,
  Input,
} from '@fluentui/react-components';
import { AddRegular, EditRegular, DeleteRegular } from '@fluentui/react-icons';
import { RoomType } from '../../types';
import { roomTypesService } from '../../services/roomTypesService';

const useStyles = makeStyles({
  root: {},
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  pageTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 2.5vw, 28px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.02em',
  },
  addBtn: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '13px',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: '8px 18px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    transition: 'background 0.2s',
  },
  tableWrap: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    overflow: 'auto',
  },
  table: { width: '100%', borderCollapse: 'collapse' as const },
  th: {
    textAlign: 'left' as const,
    padding: '11px 14px',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '700',
    letterSpacing: '0.04em',
    textTransform: 'uppercase' as const,
  },
  td: {
    padding: '11px 14px',
    borderBottom: '1px solid rgba(20,27,77,0.07)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    color: 'var(--color-text)',
    verticalAlign: 'middle' as const,
  },
  actions: { display: 'flex', gap: '6px' },
  actionBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '600',
    border: '1.5px solid rgba(20,27,77,0.25)',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 10px',
    cursor: 'pointer',
    background: 'transparent',
    color: 'var(--color-primary)',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  deleteBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '600',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 10px',
    cursor: 'pointer',
    background: 'transparent',
    color: '#b71c1c',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  formGrid: { display: 'flex', flexDirection: 'column', gap: '14px' },
  hint: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    color: 'rgba(20,27,77,0.5)',
    marginTop: '2px',
  },
  empty: {
    padding: '32px',
    textAlign: 'center' as const,
    fontFamily: 'var(--font-brand)',
    color: 'rgba(20,27,77,0.45)',
    fontSize: '14px',
  },
});

interface TypeForm { name: string; label: string; }
const emptyForm = (): TypeForm => ({ name: '', label: '' });

export function RoomTypesPage() {
  const styles = useStyles();
  const [types, setTypes] = useState<RoomType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<RoomType | null>(null);
  const [form, setForm] = useState<TypeForm>(emptyForm());

  const load = () => {
    roomTypesService.getAll()
      .then(setTypes)
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };
  useEffect(() => { load(); }, []);

  const openCreate = () => { setEditing(null); setForm(emptyForm()); setDialogOpen(true); };
  const openEdit = (t: RoomType) => {
    setEditing(t);
    setForm({ name: t.name, label: t.label });
    setDialogOpen(true);
  };

  const handleSave = async () => {
    try {
      if (editing) {
        await roomTypesService.update(editing.id, form);
      } else {
        await roomTypesService.create(form);
      }
      setDialogOpen(false);
      load();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: string } })?.response?.data;
      alert(typeof msg === 'string' ? msg : 'Помилка збереження.');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Видалити тип приміщення?')) return;
    try {
      await roomTypesService.delete(id);
      setTypes((prev) => prev.filter((t) => t.id !== id));
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: string } })?.response?.data;
      alert(typeof msg === 'string' ? msg : 'Помилка видалення.');
    }
  };

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.pageTitle}>Типи приміщень</span>
        <button
          className={styles.addBtn}
          onClick={openCreate}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-secondary-2)')}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-primary)')}
        >
          <AddRegular fontSize={16} />
          Додати тип
        </button>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження..." />
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th className={styles.th}>Ярлик</th>
                <th className={styles.th}>Назва</th>
                <th className={styles.th}>Дії</th>
              </tr>
            </thead>
            <tbody>
              {types.length === 0 ? (
                <tr><td colSpan={3} className={styles.empty}>Типи відсутні</td></tr>
              ) : (
                types.map((t) => (
                  <tr key={t.id}>
                    <td className={styles.td}>
                      <Badge appearance="outline" style={{ fontFamily: 'var(--font-brand)', fontSize: '12px' }}>
                        {t.name}
                      </Badge>
                    </td>
                    <td className={styles.td} style={{ fontWeight: '700', color: 'var(--color-primary)' }}>
                      {t.label}
                    </td>
                    <td className={styles.td}>
                      <div className={styles.actions}>
                        <button className={styles.actionBtn} onClick={() => openEdit(t)}>
                          <EditRegular fontSize={13} /> Ред.
                        </button>
                        <button className={styles.deleteBtn} onClick={() => handleDelete(t.id)}>
                          <DeleteRegular fontSize={13} /> Вид.
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      <Dialog open={dialogOpen} onOpenChange={(_, d) => setDialogOpen(d.open)}>
        <DialogSurface>
          <DialogBody>
            <DialogTitle style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)', fontWeight: '700' }}>
              {editing ? 'Редагувати тип' : 'Новий тип приміщення'}
            </DialogTitle>
            <DialogContent>
              <div className={styles.formGrid}>
                <Field label="Ярлик" required hint="Латиниця, в нижньому регістрі, без пробілів: classroom, sport, conference">
                  <Input
                    value={form.name}
                    onChange={(_, d) => setForm((f) => ({ ...f, name: d.value }))}
                    placeholder="classroom"
                    style={{ fontFamily: 'var(--font-brand)' }}
                  />
                </Field>
                <Field label="Назва для відображення" required>
                  <Input
                    value={form.label}
                    onChange={(_, d) => setForm((f) => ({ ...f, label: d.value }))}
                    placeholder="Аудиторія"
                    style={{ fontFamily: 'var(--font-brand)' }}
                  />
                </Field>
              </div>
            </DialogContent>
            <DialogActions>
              <DialogTrigger disableButtonEnhancement>
                <button style={{ fontFamily: 'var(--font-brand)', fontSize: '13px', background: 'transparent', border: '1.5px solid rgba(20,27,77,0.25)', borderRadius: 'var(--radius-md)', padding: '7px 16px', cursor: 'pointer', color: 'var(--color-primary)' }}>
                  Скасувати
                </button>
              </DialogTrigger>
              <button
                style={{ fontFamily: 'var(--font-brand)', fontSize: '13px', fontWeight: '700', background: 'var(--color-primary)', color: 'var(--color-white)', border: 'none', borderRadius: 'var(--radius-md)', padding: '7px 16px', cursor: 'pointer' }}
                onClick={handleSave}
              >
                Зберегти
              </button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>
    </div>
  );
}
