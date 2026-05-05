import React, { useEffect, useState } from 'react';
import {
  makeStyles,
  tokens,
  Text,
  SearchBox,
  Spinner,
  Tab,
  TabList,
} from '@fluentui/react-components';
import { Room, RoomType } from '../types';
import { roomsService } from '../services/roomsService';
import { roomTypesService } from '../services/roomTypesService';
import { RoomCard } from '../components/shared/RoomCard';
import heroBg from '../assets/images/hero-main-building.png';

const useStyles = makeStyles({
  root: {
    maxWidth: '1200px',
    margin: '0 auto',
  },
  hero: {
    position: 'relative',
    borderRadius: 'var(--radius-lg)',
    overflow: 'hidden',
    marginBottom: '40px',
    minHeight: '260px',
    display: 'flex',
    alignItems: 'flex-end',
    boxShadow: 'var(--shadow-card)',
  },
  heroBg: {
    position: 'absolute',
    inset: '0',
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    objectPosition: 'center 30%',
    display: 'block',
  },
  heroOverlay: {
    position: 'absolute',
    inset: '0',
    background: 'linear-gradient(to top, rgba(20,27,77,0.90) 0%, rgba(20,27,77,0.45) 55%, rgba(20,27,77,0.10) 100%)',
  },
  heroContent: {
    position: 'relative',
    zIndex: '1',
    padding: '32px 36px',
    width: '100%',
    '@media (max-width: 640px)': {
      padding: '20px 18px',
    },
  },
  heroTitle: {
    fontFamily: 'var(--font-brand)',
    color: 'var(--color-white)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 3vw, 34px)',
    letterSpacing: '0.02em',
    marginBottom: '6px',
    lineHeight: '1.2',
    display: 'block',
  },
  heroSub: {
    fontFamily: 'var(--font-brand)',
    color: 'rgba(255,255,255,0.78)',
    fontSize: 'clamp(13px, 1.5vw, 16px)',
    display: 'block',
  },
  heroMotto: {
    fontFamily: 'var(--font-brand)',
    color: 'rgba(255,255,255,0.50)',
    fontSize: '12px',
    fontStyle: 'italic',
    marginTop: '10px',
    display: 'block',
  },
  sectionTitle: {
    fontFamily: 'var(--font-brand)',
    color: 'var(--color-primary)',
    fontWeight: '700',
    fontSize: '20px',
    letterSpacing: '0.02em',
    marginBottom: '16px',
    display: 'block',
  },
  filters: {
    display: 'flex',
    gap: '12px',
    marginBottom: '20px',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  tabList: {
    borderBottom: '2px solid rgba(20, 27, 77, 0.10)',
    marginBottom: '20px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '20px',
    '@media (max-width: 640px)': {
      gridTemplateColumns: '1fr',
      gap: '14px',
    },
  },
  empty: {
    textAlign: 'center',
    padding: '60px 0',
    color: tokens.colorNeutralForeground3,
    fontFamily: 'var(--font-brand)',
  },
});

export function Home() {
  const styles = useStyles();
  const [rooms, setRooms] = useState<Room[]>([]);
  const [roomTypes, setRoomTypes] = useState<RoomType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState<string>('all');

  useEffect(() => {
    Promise.all([
      roomsService.getAll(),
      roomTypesService.getAll(),
    ])
      .then(([r, rt]) => { setRooms(r); setRoomTypes(rt); })
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, []);

  const filtered = rooms.filter((r) => {
    if (!r.isActive) return false;
    const q = search.toLowerCase();
    const matchSearch =
      !q ||
      r.name.toLowerCase().includes(q) ||
      (r.roomNumber ?? '').toLowerCase().includes(q) ||
      (r.description ?? '').toLowerCase().includes(q);
    const matchTab =
      tab === 'all' || r.roomType?.name === tab;
    return matchSearch && matchTab;
  });

  return (
    <div className={styles.root}>
      <div className={styles.hero}>
        <img src={heroBg} alt="Головний корпус Львівського університету" className={styles.heroBg} />
        <div className={styles.heroOverlay} />
        <div className={styles.heroContent}>
          <span className={styles.heroTitle}>Система бронювання приміщень</span>
          <span className={styles.heroSub}>
            Lviv University — резервування аудиторій та спортивних залів онлайн
          </span>
          <span className={styles.heroMotto}>
            «Patriae decori civibus educandis»
          </span>
        </div>
      </div>

      <Text className={styles.sectionTitle} as="h2">Доступні приміщення</Text>

      <div className={styles.filters}>
        <SearchBox
          placeholder="Пошук за назвою або номером..."
          value={search}
          onChange={(_, d) => setSearch(d.value)}
          style={{ minWidth: '240px', fontFamily: 'var(--font-brand)' }}
        />
      </div>

      <div className={styles.tabList}>
        <TabList
          selectedValue={tab}
          onTabSelect={(_, d) => setTab(d.value as string)}
          style={{ fontFamily: 'var(--font-brand)' }}
        >
          <Tab value="all" style={{ fontFamily: 'var(--font-brand)', fontWeight: tab === 'all' ? '700' : '400' }}>
            Всі
          </Tab>
          {roomTypes.map((rt) => (
            <Tab key={rt.id} value={rt.name} style={{ fontFamily: 'var(--font-brand)', fontWeight: tab === rt.name ? '700' : '400' }}>
              {rt.label}
            </Tab>
          ))}
        </TabList>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження приміщень..." />
      ) : filtered.length === 0 ? (
        <div className={styles.empty}>
          <Text size={400} style={{ fontFamily: 'var(--font-brand)' }}>Приміщень не знайдено</Text>
        </div>
      ) : (
        <div className={styles.grid}>
          {filtered.map((room) => (
            <RoomCard key={room.id} room={room} />
          ))}
        </div>
      )}
    </div>
  );
}
