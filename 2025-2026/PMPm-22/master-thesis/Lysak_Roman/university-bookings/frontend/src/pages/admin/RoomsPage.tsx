import React, { useEffect, useState, useMemo } from 'react';
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
  Textarea,
  Switch,
  Select,
} from '@fluentui/react-components';
import { AddRegular, EditRegular, DeleteRegular, SearchRegular, ChevronRightRegular } from '@fluentui/react-icons';
import { Room, RoomType, Address, StaffMember, SlotMode, UNIVERSITY_PARA } from '../../types';
import { roomsService } from '../../services/roomsService';
import { roomTypesService } from '../../services/roomTypesService';
import { addressesService } from '../../services/addressesService';
import api from '../../services/api';
import { Pagination } from '../../components/shared/Pagination';

const PAGE_SIZE = 10;

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
  topBar: {
    marginBottom: '20px',
  },
  searchWrap: {
    maxWidth: '340px',
  },
  tableWrap: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    overflow: 'auto',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
  },
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
    transition: 'background 0.15s',
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
    transition: 'background 0.15s',
  },
  formGrid: { display: 'flex', flexDirection: 'column', gap: '14px' },
  empty: {
    padding: '32px',
    textAlign: 'center' as const,
    fontFamily: 'var(--font-brand)',
    color: 'rgba(20,27,77,0.45)',
    fontSize: '14px',
  },
  clickableRow: {
    cursor: 'pointer',
    ':hover': { background: 'rgba(20,27,77,0.03)' },
  },
  expandedTd: {
    padding: '0',
    borderBottom: '1px solid rgba(20,27,77,0.07)',
  },
  expandedPanel: {
    padding: '16px 20px 20px',
    background: 'rgba(20,27,77,0.025)',
    borderTop: '1px solid rgba(20,27,77,0.06)',
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    columnGap: '32px',
    rowGap: '14px',
  },
  detailLabel: {
    fontFamily: 'var(--font-brand)',
    fontSize: '10px',
    fontWeight: '700',
    color: 'rgba(20,27,77,0.45)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    marginBottom: '3px',
  },
  detailValue: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    color: 'var(--color-text)',
    whiteSpace: 'pre-wrap' as const,
  },
  availChip: {
    fontFamily: 'var(--font-brand)',
    fontSize: '11px',
    fontWeight: '600',
    background: 'rgba(20,27,77,0.08)',
    color: 'var(--color-primary)',
    borderRadius: 'var(--radius-sm)',
    padding: '3px 10px',
  },
});

const DAY_NAMES = ['Нд', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'];

function TimeInput({ value, onChange, style }: { value: string; onChange: (v: string) => void; style?: React.CSSProperties }) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let v = e.target.value.replace(/[^0-9:]/g, '');
    if (v.length === 2 && !v.includes(':')) v = v + ':';
    if (v.length > 5) v = v.slice(0, 5);
    onChange(v);
  };
  return (
    <input
      type="text"
      value={value}
      onChange={handleChange}
      placeholder="ГГ:ХХ"
      maxLength={5}
      style={style}
    />
  );
}

interface RoomForm {
  name: string;
  roomNumber: string;
  roomTypeId: string;
  addressId: string;
  description: string;
  capacity: string;
  isActive: boolean;
  responsiblePersonId: string;
  slotMode: SlotMode;
}

interface PendingAvailability {
  key: number;
  dayOfWeek: number;
  startTime: string;
  endTime: string;
  availableParaIndices: number[];
}

const emptyForm = (): RoomForm => ({
  name: '',
  roomNumber: '',
  roomTypeId: '',
  addressId: '',
  description: '',
  capacity: '',
  isActive: true,
  responsiblePersonId: '',
  slotMode: SlotMode.Interval,
});

export function RoomsPage() {
  const styles = useStyles();
  const [rooms, setRooms] = useState<Room[]>([]);
  const [staff, setStaff] = useState<StaffMember[]>([]);
  const [roomTypes, setRoomTypes] = useState<RoomType[]>([]);
  const [addresses, setAddresses] = useState<Address[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<Room | null>(null);
  const [form, setForm] = useState<RoomForm>(emptyForm());
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // availability editor inside create/edit dialog
  const [dialogAvailDay, setDialogAvailDay] = useState('1');
  const [dialogAvailStart, setDialogAvailStart] = useState('08:00');
  const [dialogAvailEnd, setDialogAvailEnd] = useState('20:00');
  const [dialogParaIndices, setDialogParaIndices] = useState<number[]>([1, 2, 3, 4, 5, 6, 7, 8]);
  // pending slots for new rooms (collected before the room exists)
  const [pendingAvailabilities, setPendingAvailabilities] = useState<PendingAvailability[]>([]);

  const load = () => {
    Promise.all([
      roomsService.adminGetAll(),
      api.get<StaffMember[]>('/api/admin/staff').then((r) => r.data),
      roomTypesService.getAll(),
      addressesService.getAll(),
    ])
      .then(([rms, st, rt, addr]) => {
        setRooms(rms);
        setStaff(st);
        setRoomTypes(rt);
        setAddresses(addr);
      })
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };
  useEffect(() => { load(); }, []);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return rooms;
    return rooms.filter(
      (r) =>
        r.name.toLowerCase().includes(q) ||
        (r.roomNumber ?? '').toLowerCase().includes(q) ||
        (r.roomType?.label ?? '').toLowerCase().includes(q),
    );
  }, [rooms, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const handleSearch = (val: string) => {
    setSearchQuery(val);
    setCurrentPage(1);
  };

  const openCreate = () => {
    setEditing(null);
    setForm(emptyForm());
    setPendingAvailabilities([]);
    setDialogAvailDay('1');
    setDialogAvailStart('08:00');
    setDialogAvailEnd('20:00');
    setDialogParaIndices([1, 2, 3, 4, 5, 6, 7, 8]);
    setDialogOpen(true);
  };
  const openEdit = (r: Room) => {
    setEditing(r);
    setForm({
      name: r.name,
      roomNumber: r.roomNumber ?? '',
      roomTypeId: r.roomType?.id ?? '',
      addressId: r.address?.id ?? '',
      description: r.description ?? '',
      capacity: r.capacity != null ? String(r.capacity) : '',
      isActive: r.isActive,
      responsiblePersonId: r.responsiblePerson?.id ?? '',
      slotMode: r.slotMode ?? SlotMode.Interval,
    });
    setPendingAvailabilities([]);
    setDialogAvailDay('1');
    setDialogAvailStart('08:00');
    setDialogAvailEnd('20:00');
    setDialogParaIndices([1, 2, 3, 4, 5, 6, 7, 8]);
    setDialogOpen(true);
  };

  const handleSave = async () => {
    if (!form.name.trim()) {
      alert('Назва є обов\'язковим полем.');
      return;
    }

    const currentSlots = editing
      ? (editing.availability ?? [])
      : pendingAvailabilities;

    if (form.isActive && currentSlots.length === 0) {
      alert('Активне приміщення повинно мати хоча б один день розкладу доступності.');
      return;
    }

    const payload = {
      name: form.name.trim(),
      roomNumber: form.roomNumber || undefined,
      roomTypeId: form.roomTypeId || undefined,
      addressId: form.addressId || undefined,
      description: form.description || undefined,
      capacity: form.capacity ? parseInt(form.capacity) : undefined,
      isActive: form.isActive,
      responsiblePersonId: form.responsiblePersonId || undefined,
      slotMode: form.slotMode,
    };
    try {
      if (editing) {
        await roomsService.adminUpdate(editing.id, payload);
      } else {
        const created = await roomsService.adminCreate(payload);
        if (pendingAvailabilities.length > 0 && created?.id) {
          for (const slot of pendingAvailabilities) {
            await roomsService.adminAddAvailability(created.id, {
              dayOfWeek: slot.dayOfWeek,
              startTime: slot.startTime,
              endTime: slot.endTime,
              availableParaIndices: slot.availableParaIndices,
            });
          }
        }
      }
      setDialogOpen(false);
      load();
    } catch { alert('Помилка збереження.'); load(); }
  };

  const handleDialogAddAvail = async () => {
    const isPara = form.slotMode === SlotMode.Para;
    const startTime = isPara ? '08:30' : dialogAvailStart;
    const endTime = isPara ? '21:00' : dialogAvailEnd;
    const paraIndices = isPara
      ? (dialogParaIndices.length === 8 ? [] : dialogParaIndices)
      : [];

    if (!isPara) {
      const timeRegex = /^([01]\d|2[0-3]):([0-5]\d)$/;
      if (!timeRegex.test(startTime) || !timeRegex.test(endTime)) {
        alert('Введіть час у форматі ГГ:ХХ (наприклад, 08:00).');
        return;
      }
      if (startTime >= endTime) {
        alert('Час початку повинен бути меншим за час закінчення.');
        return;
      }
    }

    const dayNum = parseInt(dialogAvailDay);

    if (editing) {
      const duplicate = (editing.availability ?? []).some((av) => av.dayOfWeek === dayNum);
      if (duplicate) {
        alert(`День "${DAY_NAMES[dayNum]}" вже є в розкладі. Видаліть існуючий запис, щоб замінити його.`);
        return;
      }
      try {
        await roomsService.adminAddAvailability(editing.id, {
          dayOfWeek: dayNum,
          startTime,
          endTime,
          availableParaIndices: paraIndices,
        });
        const updated = await roomsService.adminGetAll();
        setRooms(updated);
        setEditing(updated.find((r) => r.id === editing.id) ?? null);
      } catch { alert('Помилка додавання розкладу.'); }
    } else {
      const duplicate = pendingAvailabilities.some((av) => av.dayOfWeek === dayNum);
      if (duplicate) {
        alert(`День "${DAY_NAMES[dayNum]}" вже є в розкладі. Видаліть існуючий запис, щоб замінити його.`);
        return;
      }
      setPendingAvailabilities((prev) => [
        ...prev,
        { key: Date.now(), dayOfWeek: dayNum, startTime, endTime, availableParaIndices: paraIndices },
      ]);
    }
    setDialogParaIndices([1, 2, 3, 4, 5, 6, 7, 8]);
  };

  const handleDialogDeleteAvail = async (availId: string) => {
    if (!editing) return;
    try {
      await roomsService.adminDeleteAvailability(editing.id, availId);
      const updated = await roomsService.adminGetAll();
      setRooms(updated);
      setEditing(updated.find((r) => r.id === editing.id) ?? null);
    } catch { alert('Помилка видалення розкладу.'); }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Видалити приміщення?')) return;
    try {
      await roomsService.adminDelete(id);
      setRooms((prev) => prev.filter((r) => r.id !== id));
    } catch { alert('Помилка видалення.'); }
  };

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.pageTitle}>Приміщення</span>
        <button
          className={styles.addBtn}
          onClick={openCreate}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-secondary-2)')}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-primary)')}
        >
          <AddRegular fontSize={16} />
          Додати приміщення
        </button>
      </div>

      <div className={styles.topBar}>
        <div className={styles.searchWrap}>
          <Input
            placeholder="Пошук за назвою або типом..."
            value={searchQuery}
            onChange={(_, d) => handleSearch(d.value)}
            contentBefore={<SearchRegular fontSize={16} />}
            style={{ fontFamily: 'var(--font-brand)', width: '100%' }}
          />
        </div>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження..." />
      ) : (
        <>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Назва</th>
                  <th className={styles.th}>Тип</th>
                  <th className={styles.th}>Слоти</th>
                  <th className={styles.th}>Місткість</th>
                  <th className={styles.th}>Статус</th>
                  <th className={styles.th}>Дії</th>
                </tr>
              </thead>
              <tbody>
                {paginated.length === 0 ? (
                  <tr>
                    <td colSpan={6} className={styles.empty}>Нічого не знайдено</td>
                  </tr>
                ) : (
                  paginated.map((r) => {
                    const isExpanded = expandedId === r.id;
                    const sortedAvail = (r.availability ?? []).slice().sort((a, b) => a.dayOfWeek - b.dayOfWeek);
                    return (
                      <React.Fragment key={r.id}>
                        <tr
                          className={styles.clickableRow}
                          style={{ background: isExpanded ? 'rgba(20,27,77,0.03)' : undefined }}
                          onClick={() => setExpandedId(isExpanded ? null : r.id)}
                        >
                          <td className={styles.td} style={{ fontWeight: '700', color: 'var(--color-primary)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <ChevronRightRegular
                                fontSize={14}
                                style={{
                                  transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                                  transition: 'transform 0.2s',
                                  flexShrink: 0,
                                  color: 'rgba(20,27,77,0.4)',
                                }}
                              />
                              {r.name}
                            </div>
                          </td>
                          <td className={styles.td}>{r.roomType?.label ?? '—'}</td>
                          <td className={styles.td}>{r.slotMode === SlotMode.Para ? 'Пари' : 'Інтервал'}</td>
                          <td className={styles.td}>{r.capacity != null ? `${r.capacity} осіб` : '—'}</td>
                          <td className={styles.td}>
                            <Badge color={r.isActive ? 'success' : 'danger'}>
                              {r.isActive ? 'Активне' : 'Неактивне'}
                            </Badge>
                          </td>
                          <td className={styles.td} onClick={(e) => e.stopPropagation()}>
                            <div className={styles.actions}>
                              <button className={styles.actionBtn} onClick={() => openEdit(r)}>
                                <EditRegular fontSize={13} /> Ред.
                              </button>
                              <button className={styles.deleteBtn} onClick={() => handleDelete(r.id)}>
                                <DeleteRegular fontSize={13} /> Вид.
                              </button>
                            </div>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={6} className={styles.expandedTd}>
                              <div className={styles.expandedPanel}>
                                <div>
                                  <div className={styles.detailLabel}>Номер кімнати</div>
                                  <div className={styles.detailValue}>{r.roomNumber ?? '—'}</div>
                                </div>
                                <div>
                                  <div className={styles.detailLabel}>Опис</div>
                                  <div className={styles.detailValue}>{r.description || '—'}</div>
                                </div>
                                <div>
                                  <div className={styles.detailLabel}>Адреса</div>
                                  <div className={styles.detailValue}>{r.address?.street ?? '—'}</div>
                                </div>
                                <div>
                                  <div className={styles.detailLabel}>Відповідальна особа</div>
                                  <div className={styles.detailValue}>{r.responsiblePerson?.displayName ?? '—'}</div>
                                </div>
                                <div style={{ gridColumn: '1 / -1' }}>
                                  <div className={styles.detailLabel}>Розклад доступності</div>
                                  {sortedAvail.length > 0 ? (
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                                      {sortedAvail.map((av) => (
                                        <span key={'id' in av ? av.id : av.dayOfWeek} className={styles.availChip}>
                                          {r.slotMode === SlotMode.Para
                                            ? `${DAY_NAMES[av.dayOfWeek]}${av.availableParaIndices?.length > 0 ? ` · ${av.availableParaIndices.join(', ')} пара` : ''}`
                                            : `${DAY_NAMES[av.dayOfWeek]} · ${av.startTime}–${av.endTime}`}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <div className={styles.detailValue}>—</div>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            totalItems={filtered.length}
            pageSize={PAGE_SIZE}
            onPageChange={setCurrentPage}
          />
        </>
      )}

      <Dialog open={dialogOpen} onOpenChange={(_, d) => setDialogOpen(d.open)}>
        <DialogSurface>
          <DialogBody>
            <DialogTitle style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)', fontWeight: '700' }}>
              {editing ? 'Редагувати приміщення' : 'Нове приміщення'}
            </DialogTitle>
            <DialogContent style={{ maxHeight: '70vh', overflowY: 'auto' }}>
              <div className={styles.formGrid}>
                <Field label="Назва" required>
                  <Input value={form.name} onChange={(_, d) => setForm((f) => ({ ...f, name: d.value }))} style={{ fontFamily: 'var(--font-brand)' }} />
                </Field>
                <Field label="Номер кімнати">
                  <Input value={form.roomNumber} onChange={(_, d) => setForm((f) => ({ ...f, roomNumber: d.value }))} style={{ fontFamily: 'var(--font-brand)' }} />
                </Field>
                <Field label="Тип приміщення">
                  <Select value={form.roomTypeId} onChange={(_, d) => setForm((f) => ({ ...f, roomTypeId: d.value }))} style={{ fontFamily: 'var(--font-brand)' }}>
                    <option value="">Не вказано</option>
                    {roomTypes.map((rt) => (
                      <option key={rt.id} value={rt.id}>{rt.label}</option>
                    ))}
                  </Select>
                </Field>
                <Field label="Адреса">
                  <Select value={form.addressId} onChange={(_, d) => setForm((f) => ({ ...f, addressId: d.value }))} style={{ fontFamily: 'var(--font-brand)' }}>
                    <option value="">Не вказано</option>
                    {addresses.map((a) => (
                      <option key={a.id} value={a.id}>{a.street}</option>
                    ))}
                  </Select>
                </Field>
                <Field label="Опис">
                  <Textarea value={form.description} onChange={(_, d) => setForm((f) => ({ ...f, description: d.value }))} rows={3} style={{ fontFamily: 'var(--font-brand)' }} />
                </Field>
                <Field label="Місткість (осіб)">
                  <Input type="number" min="0" value={form.capacity} onChange={(_, d) => setForm((f) => ({ ...f, capacity: d.value }))} style={{ fontFamily: 'var(--font-brand)' }} />
                </Field>
                <Field label="Відповідальна особа">
                  <Select value={form.responsiblePersonId} onChange={(_, d) => setForm((f) => ({ ...f, responsiblePersonId: d.value }))} style={{ fontFamily: 'var(--font-brand)' }}>
                    <option value="">Не призначено</option>
                    {staff.map((s) => <option key={s.id} value={s.id}>{s.displayName}</option>)}
                  </Select>
                </Field>
                <Switch label="Активне" checked={form.isActive} onChange={(_, d) => setForm((f) => ({ ...f, isActive: d.checked }))} />

                <Field label="Режим слотів">
                  <Select
                    value={form.slotMode}
                    onChange={(_, d) => setForm((f) => ({ ...f, slotMode: d.value as SlotMode }))}
                    style={{ fontFamily: 'var(--font-brand)' }}
                  >
                    <option value={SlotMode.Interval}>Інтервальні слоти</option>
                    <option value={SlotMode.Para}>Пари</option>
                  </Select>
                </Field>

                {/* Availability inside dialog */}
                <div style={{ borderTop: '1.5px solid rgba(20,27,77,0.1)', paddingTop: '14px', marginTop: '4px' }}>
                  <span style={{ fontFamily: 'var(--font-brand)', fontWeight: '700', fontSize: '13px', color: 'var(--color-primary)', display: 'block', marginBottom: '10px' }}>
                    Розклад доступності
                  </span>

                  {/* existing slots (edit mode) or pending slots (create mode) */}
                  {(() => {
                    const slots = editing
                      ? (editing.availability ?? []).slice().sort((a, b) => a.dayOfWeek - b.dayOfWeek)
                      : pendingAvailabilities;
                    if (slots.length === 0) {
                      return (
                        <span style={{ fontFamily: 'var(--font-brand)', fontSize: '12px', color: 'rgba(20,27,77,0.45)', display: 'block', marginBottom: '10px' }}>
                          Розклад не налаштовано.
                        </span>
                      );
                    }
                    return (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '12px' }}>
                        {slots.map((av) => (
                          <div
                            key={'id' in av ? av.id : av.key}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              background: 'rgba(20,27,77,0.04)',
                              borderRadius: 'var(--radius-sm)',
                              padding: '6px 12px',
                              fontFamily: 'var(--font-brand)',
                              fontSize: '13px',
                              color: 'var(--color-primary)',
                            }}
                          >
                            <span>
                              {DAY_NAMES[av.dayOfWeek]}
                              {form.slotMode === SlotMode.Interval && ` · ${'startTime' in av ? av.startTime : ''} – ${'endTime' in av ? av.endTime : ''}`}
                              {form.slotMode === SlotMode.Para && (() => {
                                const indices = 'availableParaIndices' in av ? av.availableParaIndices : [];
                                return indices.length > 0 ? ` · ${indices.join(', ')} пара` : '';
                              })()}
                            </span>
                            <button
                              type="button"
                              onClick={() => {
                                if (editing && 'id' in av) {
                                  handleDialogDeleteAvail(av.id);
                                } else {
                                  setPendingAvailabilities((prev) => prev.filter((p) => p.key !== (av as PendingAvailability).key));
                                }
                              }}
                              style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#b71c1c', fontFamily: 'var(--font-brand)', fontSize: '13px', padding: '2px 6px' }}
                            >
                              ✕
                            </button>
                          </div>
                        ))}
                      </div>
                    );
                  })()}

                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    <Field label="День тижня">
                      <Select value={dialogAvailDay} onChange={(_, d) => setDialogAvailDay(d.value)} style={{ fontFamily: 'var(--font-brand)' }}>
                        {DAY_NAMES.map((name, i) => (
                          <option key={i} value={i}>{name}</option>
                        ))}
                      </Select>
                    </Field>
                    {form.slotMode === SlotMode.Interval && (
                      <>
                        <Field label="Від">
                          <TimeInput value={dialogAvailStart} onChange={setDialogAvailStart} style={{ fontFamily: 'var(--font-brand)', padding: '5px 8px', border: '1px solid rgba(20,27,77,0.3)', borderRadius: 'var(--radius-sm)', fontSize: '14px', color: 'var(--color-text)', width: '72px' }} />
                        </Field>
                        <Field label="До">
                          <TimeInput value={dialogAvailEnd} onChange={setDialogAvailEnd} style={{ fontFamily: 'var(--font-brand)', padding: '5px 8px', border: '1px solid rgba(20,27,77,0.3)', borderRadius: 'var(--radius-sm)', fontSize: '14px', color: 'var(--color-text)', width: '72px' }} />
                        </Field>
                      </>
                    )}
                    <button
                      type="button"
                      onClick={handleDialogAddAvail}
                      style={{ fontFamily: 'var(--font-brand)', fontWeight: '700', fontSize: '12px', backgroundColor: 'var(--color-primary)', color: 'var(--color-white)', border: 'none', borderRadius: 'var(--radius-md)', padding: '7px 14px', cursor: 'pointer', alignSelf: 'flex-end' }}
                    >
                      + Додати
                    </button>
                  </div>
                  {form.slotMode === SlotMode.Para && (
                    <div style={{ marginTop: '10px' }}>
                      <span style={{ fontFamily: 'var(--font-brand)', fontSize: '12px', fontWeight: '700', color: 'rgba(20,27,77,0.6)', display: 'block', marginBottom: '8px' }}>
                        Доступні пари для цього дня
                      </span>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {UNIVERSITY_PARA.map((p) => {
                          const checked = dialogParaIndices.includes(p.index);
                          return (
                            <label
                              key={p.index}
                              style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontFamily: 'var(--font-brand)', fontSize: '12px', color: 'var(--color-text)', background: checked ? 'rgba(20,27,77,0.07)' : 'transparent', borderRadius: 'var(--radius-sm)', padding: '4px 8px', border: '1px solid rgba(20,27,77,0.15)' }}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={(e) => {
                                  setDialogParaIndices((prev) =>
                                    e.target.checked
                                      ? [...prev, p.index].sort((a, b) => a - b)
                                      : prev.filter((i) => i !== p.index),
                                  );
                                }}
                                style={{ accentColor: 'var(--color-primary)', width: '13px', height: '13px', cursor: 'pointer' }}
                              />
                              {p.label}
                            </label>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
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
