"""Frame-local WHITE boundary label propagation (shadow only)."""
from dataclasses import dataclass
import numpy as np
from .geometry import tangents, OrderedPolyline
from .roles import Component

LEFT = 'LEFT'
RIGHT = 'RIGHT'
AMBIGUOUS = 'AMBIGUOUS'

@dataclass(frozen=True)
class WhiteShadow:
    labels: dict
    left: object = None
    right: object = None
    ambiguous: tuple = ()
    reason: str = ''
    diagnostics: dict = None

def _signed(poly, center):
    p = poly.points; c = center.points
    out=[]
    for point in p:
        j=int(np.argmin(np.linalg.norm(c-point,axis=1)))
        j=max(0,min(j,len(c)-1)); t=tangents(c)[j]; n=np.array([-t[1],t[0]])
        out.append(float(np.dot(point-c[j],n)))
    return np.asarray(out)

def _expected_distance(poly, center, half_width, sign):
    c=center.points; t=tangents(c); n=np.column_stack((-t[:,1],t[:,0]))
    expected=c + float(sign)*float(half_width)*n
    distances=np.linalg.norm(poly.points[:,None,:]-expected[None,:,:],axis=2)
    return float(np.median(np.min(distances,axis=1)))

def seed_from_center(center, whites, half_width=None, tolerance=None, previous=None):
    labels={}; left=[]; right=[]; diagnostics={}; left_candidates=[]; right_candidates=[]
    for item in whites:
        values=_signed(item.polyline, center)
        if len(values)==0: continue
        positive=float(np.mean(values>0)); negative=float(np.mean(values<0))
        median=float(np.median(values)); error=None if half_width is None else abs(abs(median)-float(half_width))
        expected_left=None if half_width is None else _expected_distance(item.polyline, center, half_width, 1.0)
        expected_right=None if half_width is None else _expected_distance(item.polyline, center, half_width, -1.0)
        diagnostics[item.component_id]=(median, error, expected_left, expected_right)
        width_ok = tolerance is None or float(tolerance) < 0 or (error is not None and error <= float(tolerance))
        if positive >= .6 and positive > negative and width_ok: left_candidates.append((expected_left if expected_left is not None else error if error is not None else 0.0, item))
        elif negative >= .6 and negative > positive and width_ok: right_candidates.append((expected_right if expected_right is not None else error if error is not None else 0.0, item))
    def key(entry, role):
        error, item = entry
        prior = previous.left if previous is not None and role == LEFT else previous.right if previous is not None else None
        continuity = 0.0 if prior is None else min(np.linalg.norm(item.polyline.points[0]-prior.polyline.points[-1]), np.linalg.norm(item.polyline.points[-1]-prior.polyline.points[-1]))
        # The current CENTER-derived expected boundary is the primary
        # reference.  Continuity is only a tie-breaker; making it primary
        # can lock an external WHITE chain after one mistaken seed.
        return (error, continuity, -item.support)
    if left_candidates:
        item=min(left_candidates,key=lambda x:key(x,LEFT))[1]; labels[item.component_id]=LEFT; left.append(_stitch_side(item,[x[1] for x in left_candidates]))
    if right_candidates:
        item=min(right_candidates,key=lambda x:key(x,RIGHT))[1]; labels[item.component_id]=RIGHT; right.append(_stitch_side(item,[x[1] for x in right_candidates]))
    # Keep one local geometry seed per side; fragmented WHITE dashes are
    # represented by the strongest observed continuation, not a persistent ID.
    return WhiteShadow(labels, max(left, key=lambda x:x.support) if left else None,
                       max(right, key=lambda x:x.support) if right else None,
                       tuple(k for k,v in labels.items() if v==AMBIGUOUS), 'CENTER_SEED', diagnostics)

def _smooth(poly):
    if len(poly.points)<4: return True
    d=np.diff(poly.points,axis=0); h=np.unwrap(np.arctan2(d[:,1],d[:,0])); dh=np.diff(h)
    significant=dh[np.abs(dh)>.35]
    return int(np.sum(np.sign(significant[1:]) != np.sign(significant[:-1]))) <= 1

def _corridor_metrics(previous, candidate):
    """Compare the whole candidate geometry with the previous boundary tail."""
    ref = previous.points
    tail = max(2, min(len(ref), 8))
    end = ref[-1]
    t = tangents(ref)[-1]
    n = np.array([-t[1], t[0]])
    pts = candidate.points
    delta = pts - end
    forward = delta @ t
    lateral = delta @ n
    # A continuation should mostly lie in front of the previous tail.
    forward_fraction = float(np.mean(forward >= -0.05))
    corridor_distance = float(np.median(np.min(np.linalg.norm(pts[:, None, :] - ref[-tail:][None, :, :], axis=2), axis=1)))
    lateral_spread = float(np.median(np.abs(lateral)))
    return forward_fraction, corridor_distance, lateral_spread

def _stitch_side(seed, candidates, gap_limit=.30):
    chain=[seed]; remaining=[x for x in candidates if x.component_id != seed.component_id]
    while remaining:
        end=chain[-1].polyline.points[-1]; choices=[]
        for item in remaining:
            p=item.polyline.points; d0=np.linalg.norm(p[0]-end); d1=np.linalg.norm(p[-1]-end)
            if min(d0,d1) <= gap_limit: choices.append((min(d0,d1),item,p if d0<=d1 else p[::-1]))
        if not choices: break
        _, item, points=min(choices,key=lambda x:x[0]); remaining.remove(item)
        chain.append(Component(item.component_id,item.color,OrderedPolyline.from_points(points),item.support))
    points=[]
    for i,item in enumerate(chain):
        p=item.polyline.points
        if i and np.linalg.norm(p[0]-points[-1])>1e-9: points.extend(np.linspace(points[-1],p[0],4)[1:-1])
        points.extend(p)
    return Component(seed.component_id,seed.color,OrderedPolyline.from_points(np.asarray(points)),sum(x.support for x in chain))

def propagate(previous, whites, max_gap=.30):
    labels={}; selected={}; used=set()
    for role, seed in ((LEFT, previous.left), (RIGHT, previous.right)):
        if seed is None: continue
        choices=[]
        all_smooth=[]
        for item in whites:
            if item.component_id in used: continue
            d=min(np.linalg.norm(item.polyline.points[0]-seed.polyline.points[-1]),
                  np.linalg.norm(item.polyline.points[-1]-seed.polyline.points[-1]))
            if _smooth(item.polyline):
                vehicle_distance=float(np.min(np.linalg.norm(item.polyline.points, axis=1)))
                forward_fraction, corridor_distance, lateral_spread = _corridor_metrics(seed.polyline, item.polyline)
                if forward_fraction < 0.5:
                    continue
                all_smooth.append((vehicle_distance, corridor_distance, lateral_spread, d, item))
                if d <= max_gap:
                    choices.append((vehicle_distance, corridor_distance, lateral_spread, d, item))
        # Vehicle proximity is primary.  The previous-seed gap is only a
        # preference; if it excludes the nearest valid WHITE, use the nearest
        # smooth observation rather than forcing a farther external chain.
        if not choices:
            choices = all_smooth
        if choices:
            item=min(choices,key=lambda x:(x[0], x[1], x[2], x[3]))[4]; labels[item.component_id]=role; selected[role]=item; used.add(item.component_id)
    collision=set()
    if LEFT in selected and RIGHT in selected and selected[LEFT].component_id == selected[RIGHT].component_id:
        collision.add(selected[LEFT].component_id); labels[selected[LEFT].component_id]=AMBIGUOUS
        selected.pop(LEFT,None); selected.pop(RIGHT,None)
    return WhiteShadow(labels, selected.get(LEFT), selected.get(RIGHT), tuple(collision), 'PROPAGATED')
