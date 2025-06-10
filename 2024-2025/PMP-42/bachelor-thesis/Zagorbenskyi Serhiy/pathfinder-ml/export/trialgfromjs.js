class Graph {
   constructor(points = [], segments = []) {
      this.points = points;
      this.segments = segments;
   }

   static load(info) {
      const points = info.points.map((i) => new Point(i.x, i.y));
      const segments = info.segments.map((i) => new Segment(
         points.find((p) => p.equals(i.p1)),
         points.find((p) => p.equals(i.p2)),
         i.oneWay
      ));
      return new Graph(points, segments);
   }

   hash() {
      return JSON.stringify(this);
   }

   addPoint(point) {
      this.points.push(point);
   }

   containsPoint(point) {
      return this.points.find((p) => p.equals(point));
   }

   tryAddPoint(point) {
      if (!this.containsPoint(point)) {
         this.addPoint(point);
         return true;
      }
      return false;
   }

   removePoint(point) {
      const segs = this.getSegmentsWithPoint(point);
      for (const seg of segs) {
         this.removeSegment(seg);
      }
      this.points.splice(this.points.indexOf(point), 1);
   }

   addSegment(seg) {
      this.segments.push(seg);
   }

   containsSegment(seg) {
      return this.segments.find((s) => s.equals(seg));
   }

   tryAddSegment(seg) {
      if (!this.containsSegment(seg) && !seg.p1.equals(seg.p2)) {
         this.addSegment(seg);
         return true;
      }
      return false;
   }

   removeSegment(seg) {
      this.segments.splice(this.segments.indexOf(seg), 1);
   }

   getSegmentsWithPoint(point) {
      const segs = [];
      for (const seg of this.segments) {
         if (seg.includes(point)) {
            segs.push(seg);
         }
      }
      return segs;
   }

   getSegmentsLeavingFromPoint(point) {
      const segs = [];
      for (const seg of this.segments) {
         if (seg.oneWay) {
            if (seg.p1.equals(point)) {
               segs.push(seg);
            }
         } else {
            if (seg.includes(point)) {
               segs.push(seg);
            }
         }
      }
      return segs;
   }



// // koloni metod!!!!



//    getShortestPath(start, end, options = {}) {
//   const {
//     antCount = 50,
//     iterations = 100,
//     alpha = 1,        // вплив феромону
//     beta = 5,         // вплив жадібності (довжина)
//     evaporation = 0.5,
//     pheromoneDeposit = 100
//   } = options;

//   let visitedCount = 0;
//   const t0 = performance.now();
//   const segments = [];
//   const pheromones = new Map();

//   // Ініціалізація сегментів і феромонів
//   for (const point of this.points) {
//     for (const seg of this.getSegmentsLeavingFromPoint(point)) {
//       const key = `${seg.p1.id}-${seg.p2.id}`;
//       if (!pheromones.has(key)) {
//         pheromones.set(key, 1); // стартовий рівень феромону
//         segments.push(seg);
//       }
//     }
//   }

//   let bestPath = null;
//   let bestLength = Infinity;

//   function getSegmentKey(p1, p2) {
//     return `${p1.id}-${p2.id}`;
//   }

//   for (let iter = 0; iter < iterations; iter++) {
//     const paths = [];

//     for (let ant = 0; ant < antCount; ant++) {
//       let current = start;
//       const visited = new Set();
//       const path = [];
//       let length = 0;

//       visited.add(current);

//       while (current !== end) {
//         const options = this.getSegmentsLeavingFromPoint(current)
//           .filter(seg => {
//             const next = seg.p1.equals(current) ? seg.p2 : seg.p1;
//             return !visited.has(next);
//           });

//         if (options.length === 0) break;

//         visitedCount++;

//         const probs = [];
//         let sum = 0;
//         for (const seg of options) {
//           const next = seg.p1.equals(current) ? seg.p2 : seg.p1;
//           const key = getSegmentKey(current, next);
//           const pher = pheromones.get(key) || 1;
//           const desirability = 1 / seg.length();
//           const prob = Math.pow(pher, alpha) * Math.pow(desirability, beta);
//           probs.push({ seg, next, prob });
//           sum += prob;
//         }

//         if (sum === 0) break;

//         // Рулетка
//         let r = Math.random() * sum;
//         let chosen = null;
//         for (const option of probs) {
//           r -= option.prob;
//           if (r <= 0) {
//             chosen = option;
//             break;
//           }
//         }

//         if (!chosen) chosen = probs[probs.length - 1];

//         path.push(chosen.seg);
//         visited.add(chosen.next);
//         length += chosen.seg.length();
//         current = chosen.next;
//       }

//       if (current === end) {
//         paths.push({ path, length });
//         if (length < bestLength) {
//           bestLength = length;
//           bestPath = [...path];
//         }
//       }
//     }

//     // Оновлення феромонів
//     for (const key of pheromones.keys()) {
//       pheromones.set(key, pheromones.get(key) * (1 - evaporation));
//     }

//     for (const { path, length } of paths) {
//       const delta = pheromoneDeposit / length;
//       for (const seg of path) {
//         const key = getSegmentKey(seg.p1, seg.p2);
//         pheromones.set(key, (pheromones.get(key) || 0) + delta);
//       }
//     }
//   }

//   // Вивід у вигляді масиву точок, як у getShortestPath
//   const pointsPath = [start];
//   if (bestPath) {
//     for (const seg of bestPath) {
//       const last = pointsPath[pointsPath.length - 1];
//       const next = seg.p1.equals(last) ? seg.p2 : seg.p1;
//       pointsPath.push(next);
//     }
//   }
//   const t1 = performance.now();
//   console.log("Time: ", t1 - t0, "ms");
//   console.log(visitedCount);
//   return pointsPath;
// }



  //deiksrta metod


  //  getShortestPath(start, end) {
  //     for (const point of this.points) {
  //        point.dist = Number.MAX_SAFE_INTEGER;
  //        point.visited = false;
  //     }

  //     const t0 = performance.now();

  //     let visitedCount = 0;
  //     let currentPoint = start;
  //     currentPoint.dist = 0;

  //     while (!end.visited) {
  //        const segs = this.getSegmentsLeavingFromPoint(currentPoint);
  //        for (const seg of segs) {
  //           const otherPoint = seg.p1.equals(currentPoint) ? seg.p2 : seg.p1;
  //           if (currentPoint.dist + seg.length() < otherPoint.dist) {
  //              otherPoint.dist = currentPoint.dist + seg.length();
  //              otherPoint.prev = currentPoint;
  //           }
  //        }


  //        if (!currentPoint.visited) {
  //         currentPoint.visited = true;
  //         visitedCount++;
  //        }

  //        const unvisited = this.points.filter((p) => p.visited == false);
  //        const dists = unvisited.map((p) => p.dist);
  //        currentPoint = unvisited.find((p) => p.dist == Math.min(...dists));
  //     }

  //     const path = [];
  //     currentPoint = end;
  //     while (currentPoint) {
  //        path.unshift(currentPoint);
  //        currentPoint = currentPoint.prev;
  //     }

  //     for (const point of this.points) {
  //        delete point.dist;
  //        delete point.visited;
  //        delete point.prev;
  //     }
  //     console.log("Visited nodes:", visitedCount);
  //     const t1 = performance.now();
  //     console.log("Time: ", t1 - t0, "ms");
  //     return path;
  //  }


  // A* metod

  getShortestPath(start, end) {

  for (const point of this.points) {
    point.g = Number.MAX_SAFE_INTEGER; // відстань від початку
    point.f = Number.MAX_SAFE_INTEGER; // g + евристика
    point.visited = false;
    point.prev = null;
  }

  let visitedCount = 0;

  const t0 = performance.now();

  start.g = 0;
  start.f = this.heuristic(start, end);

  const openSet = [start];

  while (openSet.length > 0) {
    // Знаходимо точку з мінімальним f
    let currentIndex = 0;
    for (let i = 1; i < openSet.length; i++) {
      if (openSet[i].f < openSet[currentIndex].f) {
        currentIndex = i;
      }
    }

    const currentPoint = openSet[currentIndex];

    if (currentPoint === end) {
      break;
    }

    openSet.splice(currentIndex, 1);
    if (!currentPoint.visited) {
      currentPoint.visited = true;
      visitedCount++;
    }

    const segs = this.getSegmentsLeavingFromPoint(currentPoint);
    for (const seg of segs) {
      const neighbor = seg.p1.equals(currentPoint) ? seg.p2 : seg.p1;
      if (neighbor.visited) continue;

      const tentativeG = currentPoint.g + seg.length();

      if (tentativeG < neighbor.g) {
        neighbor.prev = currentPoint;
        neighbor.g = tentativeG;
        neighbor.f = tentativeG + this.heuristic(neighbor, end);

        if (!openSet.includes(neighbor)) {
          openSet.push(neighbor);
        }
      }
    }
  }

  // Побудова шляху
  const path = [];
  let currentPoint = end;
  while (currentPoint) {
    path.unshift(currentPoint);
    currentPoint = currentPoint.prev;
  }

  // Очищення службових змінних
  for (const point of this.points) {
    delete point.g;
    delete point.f;
    delete point.visited;
    delete point.prev;
  }
  console.log("Visited nodes:", visitedCount);
  const t1 = performance.now();
  console.log("Time: ", t1 - t0, "ms");
  return path;
}

// Евристична функція: Евклідова відстань
heuristic(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}


   dispose() {
      this.points.length = 0;
      this.segments.length = 0;
   }

   draw(ctx) {
      for (const seg of this.segments) {
         seg.draw(ctx);
      }

      for (const point of this.points) {
         point.draw(ctx);
      }
   }
}