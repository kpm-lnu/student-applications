import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Badge, Text, makeStyles, tokens } from '@fluentui/react-components';
import {
  BuildingRegular,
  SportRegular,
  ConferenceRoomRegular,
  PersonRegular,
  PeopleRegular,
} from '@fluentui/react-icons';
import { Room } from '../../types';

function typeIcon(name: string | undefined): React.ReactElement {
  switch (name) {
    case 'sport':      return <SportRegular fontSize={18} />;
    case 'conference': return <ConferenceRoomRegular fontSize={18} />;
    default:           return <BuildingRegular fontSize={18} />;
  }
}

const useStyles = makeStyles({
  card: {
    width: '100%',
    cursor: 'pointer',
    background: 'var(--color-white)',
    borderLeft: '4px solid var(--color-primary)',
    boxShadow: 'var(--shadow-card)',
    borderRadius: 'var(--radius-md)',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    transition: 'transform 0.15s, box-shadow 0.15s',
    ':hover': {
      transform: 'translateY(-3px)',
      boxShadow: '0 8px 32px rgba(20, 27, 77, 0.18)',
    },
  },
  typeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: 'var(--color-secondary-3)',
    fontFamily: 'var(--font-brand)',
    fontSize: '11px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    fontWeight: '700',
  },
  name: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '16px',
    color: 'var(--color-primary)',
    letterSpacing: '0.01em',
    lineHeight: '1.3',
  },
  description: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    color: '#444',
    lineHeight: '1.5',
  },
  meta: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginTop: '4px',
  },
  metaItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    color: tokens.colorNeutralForeground3,
    fontSize: '12px',
    fontFamily: 'var(--font-brand)',
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginTop: '4px',
  },
  btn: {
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '13px',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: '8px 18px',
    cursor: 'pointer',
    transition: 'background 0.2s',
    letterSpacing: '0.02em',
  },
});

interface Props {
  room: Room;
}

export function RoomCard({ room }: Props) {
  const styles = useStyles();
  const navigate = useNavigate();

  return (
    <div className={styles.card} onClick={() => navigate(`/rooms/${room.id}`)}>
      <div className={styles.typeRow}>
        {typeIcon(room.roomType?.name)}
        {room.roomType?.label ?? '—'}
        {room.roomNumber && (
          <Badge appearance="outline" style={{ marginLeft: '6px', fontFamily: 'var(--font-brand)', fontSize: '11px' }}>
            № {room.roomNumber}
          </Badge>
        )}
      </div>

      <span className={styles.name}>{room.name}</span>

      {room.description && (
        <span className={styles.description}>{room.description}</span>
      )}

      <div className={styles.meta}>
        {room.capacity && (
          <span className={styles.metaItem}>
            <PeopleRegular fontSize={15} />
            до {room.capacity} осіб
          </span>
        )}
        {room.responsiblePerson && (
          <span className={styles.metaItem}>
            <PersonRegular fontSize={15} />
            {room.responsiblePerson.displayName}
          </span>
        )}
      </div>

      <div className={styles.footer}>
        <button
          className={styles.btn}
          onClick={(e) => {
            e.stopPropagation();
            navigate(`/rooms/${room.id}`);
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-secondary-2)')}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-primary)')}
        >
          Забронювати
        </button>
        {!room.isActive && (
          <Badge appearance="outline" color="danger">
            Недоступно
          </Badge>
        )}
      </div>
    </div>
  );
}
