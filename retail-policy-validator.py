#!/usr/bin/env python3
"""
retail_policy_validator.py  --  graph/automaton policy validator for tau2-bench RETAIL.

Audits a COMPLETED trajectory and reports whether every agent decision was *legal*
under data/tau2/domains/retail/policy.md, independent of task reward (which only
checks final DB state). Surfaces "corrupt successes": reward==1 via illegal paths.

Model = guarded labeled transition system. Each agent tool is a NODE; a tool call
is a legal EDGE firing only if all obligations hold:
  requires(auth) | cross_user | provenance(grounding) | confirm(writes) | guard(state).
Reliable codes: multiple_tool_calls, msg_and_toolcall_same_turn, action_before_auth,
ungrounded_id, cross_user_action, guard_violation (status/reason/product-type/ownership).
Heuristic code (treat as a flag, ideally re-judged by an LLM): missing_confirmation.

Usage: python retail_policy_validator.py <results.json> --db db.json [--limit N] [--show K]
"""
from __future__ import annotations
import json, re, argparse
from dataclasses import dataclass

ID_RE = re.compile(r"#W\d+|gift_card_\d+|paypal_\d+|credit_card_\d+|\b[a-z]+_[a-z]+_\d+\b|\b\d{10}\b")
# affirmative OR imperative confirmation referencing an action
AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|confirm(ed|ing)?|go ahead|please do|sounds good|"
    r"do it|that'?s right|that works|ok|okay|proceed|i confirm|approved?)\b"
    r"|\bplease (cancel|return|exchange|modify|change|update|proceed)\b"
    r"|\b(cancel|return|exchange|modify|change|update) (it|them|that|this|my order)\b", re.I)

WRITE_TOOLS = {"cancel_pending_order","modify_pending_order_address","modify_pending_order_payment",
               "modify_pending_order_items","modify_user_address","return_delivered_order_items",
               "exchange_delivered_order_items"}
NO_AUTH_OK = {"find_user_id_by_email","find_user_id_by_name_zip","list_all_product_types",
              "get_product_details","get_item_details","calculate","think"}

@dataclass
class Violation:
    code: str; tool: str; detail: str; turn: int | None = None
    def __str__(self):
        return f"[{self.code}] {self.tool}{('@t'+str(self.turn)) if self.turn is not None else ''}: {self.detail}"

class RetailEnv:
    def __init__(self, db):
        self.orders, self.users, self.item_index = {}, {}, {}
        if not db: return
        self.orders = {oid: dict(o) for oid, o in db["orders"].items()}
        self.users = db["users"]
        for pid, p in db["products"].items():
            vs = p["variants"].values() if isinstance(p["variants"], dict) else p["variants"]
            for v in vs:
                self.item_index[v["item_id"]] = {"product_id": pid, "available": v.get("available", False)}
    def status(self, oid):       o=self.orders.get(oid); return o["status"] if o else None
    def order_user(self, oid):   o=self.orders.get(oid); return o["user_id"] if o else None
    def order_item_product(self, oid, item_id):
        o=self.orders.get(oid)
        return next((it["product_id"] for it in o["items"] if it["item_id"]==item_id), None) if o else None
    def user_payment_ids(self, uid): return set((self.users.get(uid,{}).get("payment_methods") or {}).keys())
    def apply_write(self, name, args):
        oid=args.get("order_id"); o=self.orders.get(oid)
        if not o: return
        if   name=="cancel_pending_order":           o["status"]="cancelled"
        elif name=="modify_pending_order_items":     o["status"]="pending (item modified)"
        elif name=="return_delivered_order_items":   o["status"]="return requested"
        elif name=="exchange_delivered_order_items": o["status"]="exchange requested"

def _req_status(args, env, want):
    if not env.orders: return []
    st=env.status(args.get("order_id"))
    return [] if st==want else [f"status is {st!r}, requires {want!r}"]

def guard_cancel(args, env, store):
    e=_req_status(args, env, "pending")
    if args.get("reason") not in {"no longer needed","ordered by mistake"}:
        e.append(f"reason {args.get('reason')!r} not in allowed set")
    return e

def _items_guard(args, env, store, want):
    e=_req_status(args, env, want); oid=args.get("order_id")
    if env.orders and store["locked"].get(oid): e.append("order locked by prior items-modify/exchange (one-shot)")
    old,new=args.get("item_ids") or [], args.get("new_item_ids") or []
    if len(old)!=len(new): e.append(f"item_ids/new_item_ids length mismatch ({len(old)} vs {len(new)})")
    if env.item_index:
        for oi,ni in zip(old,new):
            op=env.order_item_product(oid,oi); ninfo=env.item_index.get(ni)
            if ninfo is None: e.append(f"new item {ni} unknown"); continue
            if op and ninfo["product_id"]!=op: e.append(f"item {oi}->{ni} changes product type")
            if not ninfo["available"]: e.append(f"new item {ni} not available")
    return e

def guard_modify_items(a,env,s): return _items_guard(a,env,s,"pending")
def guard_exchange(a,env,s):     return _items_guard(a,env,s,"delivered")
def guard_modify_addr(a,env,s):
    e=_req_status(a,env,"pending")
    if env.orders and s["locked"].get(a.get("order_id")): e.append("order locked (one-shot)")
    return e
def guard_modify_pay(a,env,s):
    e=_req_status(a,env,"pending"); pid=a.get("payment_method_id"); uid=env.order_user(a.get("order_id")) if env.orders else None
    if uid and pid not in env.user_payment_ids(uid): e.append(f"payment_method {pid!r} not owned by user")
    if env.orders and s["locked"].get(a.get("order_id")): e.append("order locked (one-shot)")
    return e
def guard_return(a,env,s):
    e=_req_status(a,env,"delivered"); pid=a.get("payment_method_id"); uid=env.order_user(a.get("order_id")) if env.orders else None
    if uid and pid not in env.user_payment_ids(uid): e.append(f"refund target {pid!r} not an existing payment method")
    return e

SPEC = {
 "cancel_pending_order":         dict(write=1, id_args=["order_id"], order_arg="order_id", guard=guard_cancel),
 "modify_pending_order_address": dict(write=1, id_args=["order_id"], order_arg="order_id", guard=guard_modify_addr),
 "modify_pending_order_payment": dict(write=1, id_args=["order_id","payment_method_id"], order_arg="order_id", guard=guard_modify_pay),
 "modify_pending_order_items":   dict(write=1, id_args=["order_id","item_ids","new_item_ids","payment_method_id"], order_arg="order_id", guard=guard_modify_items, locks=1),
 "modify_user_address":          dict(write=1, id_args=["user_id"], user_arg="user_id"),
 "return_delivered_order_items": dict(write=1, id_args=["order_id","item_ids","payment_method_id"], order_arg="order_id", guard=guard_return),
 "exchange_delivered_order_items":dict(write=1, id_args=["order_id","item_ids","new_item_ids","payment_method_id"], order_arg="order_id", guard=guard_exchange, locks=1),
 "get_user_details":  dict(write=0, id_args=["user_id"],  user_arg="user_id"),
 "get_order_details": dict(write=0, id_args=["order_id"], order_arg="order_id"),
 "get_product_details": dict(write=0, id_args=["product_id"]),
 "get_item_details":  dict(write=0, id_args=["item_id"]),
}

def _ids(v):
    if isinstance(v,str): return [v]
    if isinstance(v,list): return [x for x in v if isinstance(x,str)]
    return []

def check(messages, env):
    res_by_id = {m.get("id"): (m.get("content") or "", bool(m.get("error")))
                 for m in messages if m.get("role")=="tool"}
    V=[]; st={"auth":False,"bound":None,"seen":set(),"locked":{},"affirm":False}
    for m in messages:
        role=m.get("role")
        if role=="user":
            c=m.get("content") or ""
            st["affirm"]=bool(AFFIRM_RE.search(c))
            st["seen"].update(ID_RE.findall(c))          # user-provided ids are grounded
            continue
        if role=="tool":
            st["seen"].update(ID_RE.findall(m.get("content") or ""))
            continue
        if role!="assistant": continue
        tcs=m.get("tool_calls") or []; content=(m.get("content") or "").strip(); turn=m.get("turn_idx")
        if tcs and content:
            V.append(Violation("msg_and_toolcall_same_turn", tcs[0].get("name","?"), "text + tool call in one turn", turn))
        if len(tcs)>1:
            V.append(Violation("multiple_tool_calls", tcs[0].get("name","?"), f"{len(tcs)} calls in one turn (max 1)", turn))
        for c in tcs:
            name=c.get("name"); args=c.get("arguments") or {}; cid=c.get("id")
            ok_result = (not res_by_id.get(cid,("",False))[1])   # tool result not an error
            if name in ("find_user_id_by_email","find_user_id_by_name_zip") and ok_result:
                st["auth"]=True
                if st["bound"] is None: st["bound"]=res_by_id.get(cid,("",False))[0].strip()
            if name not in NO_AUTH_OK and not st["auth"]:
                V.append(Violation("action_before_auth", name, "called before successful authentication", turn))
            spec=SPEC.get(name)
            if not spec: continue
            for a in spec.get("id_args",[]):
                for tok in _ids(args.get(a)):
                    if tok not in st["seen"]:
                        V.append(Violation("ungrounded_id", name, f"{a}={tok!r} never seen in prior tool result or user msg", turn))
            uid_t = args.get(spec["user_arg"]) if spec.get("user_arg") else (env.order_user(args.get(spec["order_arg"])) if (spec.get("order_arg") and env.orders) else None)
            if st["bound"] and uid_t and uid_t!=st["bound"]:
                V.append(Violation("cross_user_action", name, f"targets {uid_t!r}, session bound to {st['bound']!r}", turn))
            if spec.get("write") and not st["affirm"]:
                V.append(Violation("missing_confirmation", name, "write not preceded by user confirmation [heuristic]", turn))
            for e in (spec.get("guard") or (lambda *a:[]))(args, env, st):
                V.append(Violation("guard_violation", name, e, turn))
            if spec.get("write") and ok_result:
                env.apply_write(name,args)
                if spec.get("locks"): st["locked"][args.get("order_id")]=True
            if spec.get("write"): st["affirm"]=False
    return V

def run(path, db_path, limit, show, strict):
    data=json.load(open(path)); sims=data["simulations"] if isinstance(data,dict) and "simulations" in data else data
    db=json.load(open(db_path)) if db_path else None
    if limit: sims=sims[:limit]
    nt=len(sims); nr1=0; nv=0; nc=0; codes={}; ex=[]
    RELIABLE={"multiple_tool_calls","msg_and_toolcall_same_turn","action_before_auth","ungrounded_id","cross_user_action","guard_violation"}
    for s in sims:
        V=check(s["messages"], RetailEnv(db))
        if strict: V=[x for x in V if x.code in RELIABLE]
        r=(s.get("reward_info") or {}).get("reward"); r1=(r in (1,1.0))
        nr1+=r1
        if V:
            nv+=1
            for x in V: codes[x.code]=codes.get(x.code,0)+1
            if r1:
                nc+=1
                if len(ex)<show: ex.append((s.get("task_id"),s.get("trial"),r,V))
    print(f"file: {path}")
    print(f"mode: {'STRICT (reliable codes only)' if strict else 'ALL codes (incl. heuristic confirmation)'}")
    print(f"simulations audited       : {nt}")
    print(f"reward==1 (task pass)     : {nr1}")
    print(f"with >=1 violation        : {nv}")
    print(f"** CORRUPT SUCCESSES **    : {nc}   (reward==1 yet policy-violating)")
    if nr1: print(f"   corrupt rate among passes: {100*nc/nr1:.1f}%")
    print("\nviolation histogram:")
    for c,n in sorted(codes.items(), key=lambda kv:-kv[1]): print(f"  {n:5d}  {c}")
    print(f"\n--- up to {show} corrupt-success examples ---")
    for tid,tr,r,V in ex:
        print(f"\n  task={tid} trial={tr} reward={r}")
        for x in V[:6]: print("    ",x)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--db",default=None)
    ap.add_argument("--limit",type=int,default=None); ap.add_argument("--show",type=int,default=4)
    ap.add_argument("--strict",action="store_true",help="report only high-precision codes (drop heuristic confirmation)")
    a=ap.parse_args(); run(a.results,a.db,a.limit,a.show,a.strict)
