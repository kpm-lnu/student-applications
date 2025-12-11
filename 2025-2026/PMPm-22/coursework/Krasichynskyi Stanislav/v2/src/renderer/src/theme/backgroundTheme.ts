import { MoveDirection, OutMode } from '@tsparticles/engine'

export const backgroundTheme = {
  fullScreen: { enable: false },
  background: { color: { value: '#222222' } },
  fpsLimit: 120,
  interactivity: {
    events: {
      onHover: { enable: true, mode: 'push' },
      onClick: { enable: true, mode: 'push' }
    },
    modes: {
      repulse: { distance: 100, duration: 0.4 },
      push: { quantity: 4 }
    }
  },
  particles: {
    number: { value: 240, density: { enable: true, area: 300 } },
    color: { value: '#00FFFF' },
    links: {
      enable: true,
      distance: 120,
      color: '#00FFFF',
      opacity: 1,
      width: 1,
      triangles: { enable: true, color: '#00FFFF', opacity: 0.1 }
    },
    move: {
      enable: true,
      speed: 2,
      direction: MoveDirection.none,
      outModes: { default: OutMode.bounce }
    },
    size: { value: { min: 1, max: 3 } },
    opacity: { value: 0.6 }
  },
  detectRetina: true
}
