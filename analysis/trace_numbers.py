"""Audit numerical traceability to same-line findings citations (stdlib only).

Checks lexical numbers and explicit arithmetic, not the semantics of a claim.
Proof paths/operands are retained for review; matching an unrelated number is not
scientific verification. No cross-line citation inheritance or unstated constants.
URLs, paths, SHA identifiers and model/dtype identifiers are not measurements.
Use --staged to audit the Git index, including cited findings, rather than working
files. Unmatched claims cause exit 1; missing staged README causes exit 2.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess

NUMBER=re.compile(r'(?<![\w.])[-+]?(?:\d+\s*\^\s*[-+]?\d+|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?:[eE][-+]?\d+)?)(?:st|nd|rd|th)?')
CITATION=re.compile(r'(?:(?:\.\./)+)?findings/[A-Za-z0-9_./-]+')
LINK=re.compile(r'(!?\[[^\]]*\])\([^\n)]*\)')
ALLOWED={'.json','.csv','.md','.txt'}


def numeric(raw):
    text=re.sub(r'(st|nd|rd|th)$','',raw).replace(',','').replace(' ','')
    if '^' in text:
        a,b=text.split('^');return Decimal(a)**int(b)
    try:return Decimal(text)
    except InvalidOperation:return None  # version strings such as 2.14.1


def visible(line):
    # Keep link labels at their original columns; hide URL/path target digits.
    line=LINK.sub(lambda m:m.group(1)+' '*(len(m.group())-len(m.group(1))),line)
    line=re.sub(r'https?://\S+|(?:\.\./)*findings/[\w./-]+',lambda m:' '*len(m.group()),line)
    line=re.sub(r'\b(?:OLMoE|Qwen\d*|DeepSeek)[\w.-]*',lambda m:' '*len(m.group()),line)
    line=re.sub(r'\b[0-9a-f]{16,}\b',lambda m:' '*len(m.group()),line,flags=re.I)
    # Pure numbers inside backticks remain claims; command/path identifiers do not.
    line=re.sub(r'`([^`]+)`',lambda m:m.group() if re.fullmatch(r'[\d.,eE+%\-^ ]+',m.group(1)) else ' '*len(m.group()),line)
    return line


def tokens(line):
    text=visible(line)
    for m in NUMBER.finditer(text):
        raw=m.group();after=text[m.end():]
        # Numeric portions of Qwen3-30B-A3B, L4, fp32 etc are identifiers.
        if after and after[0].isalpha() and not re.match(r'x(?:\b|[-–])',after):continue
        context=text[max(0,m.start()-12):m.end()+12]
        kind='reference-id' if re.search(r'arXiv\s*$',text[:m.start()],re.I) else 'number'
        if re.search(r'\d{4}-\d{2}-\d{2}',context):kind='date'
        yield dict(raw=raw,value=numeric(raw),column=m.start()+1,kind=kind,
                   percent='%' in text[m.start():m.end()+18],multiplier=bool(re.match(r'(?:[-–]\d+(?:\.\d+)?)?x\b',after)),approx=bool(re.search(r'[~≈]\s*$',text[:m.start()])))


class Repository:
    def __init__(self,root,staged=False):
        self.root=Path(root).resolve();self.staged=staged;self.cache={}
        self.index=subprocess.check_output(['git','ls-files'],cwd=self.root,text=True).splitlines() if staged else None
    def read(self,path):
        if self.staged:
            return subprocess.check_output(['git','show',':'+path],cwd=self.root,text=True,stderr=subprocess.DEVNULL)
        p=(self.root/path).resolve()
        if not p.is_relative_to(self.root):raise ValueError('citation outside repository')
        return p.read_text()
    def sources(self,citation):
        # ../ paths in spec resolve to the repository's findings/ tree.
        path=citation[citation.index('findings/'):].rstrip('.')
        if self.staged:
            return [p for p in self.index if (p==path.rstrip('/') or p.startswith(path.rstrip('/')+'/')) and Path(p).suffix in ALLOWED]
        p=self.root/path
        if not p.resolve().is_relative_to((self.root/'findings').resolve()):return []
        if p.is_dir():return [str(q.relative_to(self.root)) for q in sorted(p.rglob('*')) if q.is_file() and q.suffix in ALLOWED]
        return [path] if p.is_file() and p.suffix in ALLOWED else []


def evidence(text,path):
    """Produce numbers with exact locations; arithmetic stays inside a record."""
    values=[];groups=[]
    def add(value,location):
        if isinstance(value,bool):return
        try:d=Decimal(str(value))
        except InvalidOperation:return
        if d.is_finite():values.append((d,location))
    def walk(obj,loc):
        if isinstance(obj,dict):
            group=[]
            for key,value in obj.items():
                child=f'{loc}.{key}'
                if isinstance(value,(int,float)) and not isinstance(value,bool):group.append((Decimal(str(value)),child,key))
                walk(value,child)
            if 1<len(group)<=32:groups.append(group)
        elif isinstance(obj,list):
            if obj and len(obj)<=128 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in obj):
                nums=[Decimal(str(v)) for v in obj]
                add(statistics.median(nums),loc+' [median]')
                if re.search('tokens|flips|counts|histogram',loc,re.I):add(sum(nums),loc+' [sum]')
            for i,value in enumerate(obj):walk(value,f'{loc}[{i}]')
        elif isinstance(obj,(int,float)):add(obj,loc)
    if path.endswith('.json'):
        walk(json.loads(text),path+'$')
    elif path.endswith('.csv'):
        for rownum,row in enumerate(csv.DictReader(io.StringIO(text)),2):
            group=[]
            for key,value in row.items():
                if value is None:continue
                try:v=Decimal(value)
                except InvalidOperation:continue
                if not v.is_finite():continue
                loc=f'{path}:{rownum}:{key}';add(v,loc);group.append((v,loc,key))
            if 1<len(group)<=32:groups.append(group)
    else:
        for n,line in enumerate(text.splitlines(),1):
            for token in tokens(line):
                if token['value'] is not None and token['kind']=='number':add(token['value'],f'{path}:{n}:{token["column"]}')
    return values,groups


def matches(target,value):
    wanted=target['value']
    if wanted==value:return True
    raw=re.sub(r'(st|nd|rd|th)$','',target['raw']).replace(',','')
    # Explicit decimal/scientific notation permits rounding to its final place.
    if '.' in raw or 'e' in raw.lower():
        quantum=Decimal(1).scaleb(wanted.as_tuple().exponent)
        return abs(wanted-value)<=abs(quantum)/2
    if target['approx']:
        quantum=Decimal(10) if raw.endswith('0') else Decimal(1)
        return abs(wanted-value)<=quantum/2
    return False


def prove(target,repo,citations):
    paths=sorted({p for c in citations for p in repo.sources(c)})
    if not paths:return None,'no readable cited findings file' if citations else 'no findings citation on this line'
    for path in paths:
        if path not in repo.cache:
            try:repo.cache[path]=(repo.read(path),None)
            except (OSError,subprocess.CalledProcessError,UnicodeError):repo.cache[path]=('',([],[]))
        text,parsed=repo.cache[path]
        if target['value'] is None:
            if re.search(r'(?<![\d.])'+re.escape(target['raw'])+r'(?![\d.])',text):return f'{path}: literal {target["raw"]}',None
            continue
        if parsed is None:
            try:parsed=evidence(text,path)
            except (ValueError,csv.Error,InvalidOperation):parsed=([],[])
            repo.cache[path]=(text,parsed)
        numbers,groups=parsed
        for value,loc in numbers:
            if matches(target,value):return f'{loc} = {value}',None
            if target['percent'] and 0<=value<=1 and matches(target,100*value):return f'100 × ({loc} = {value})',None
        # Ratios/differences use two explicitly reported values in the SAME record.
        # Operand paths are exposed because arithmetic is not semantic validation.
        for group in groups:
            for a,aloc,akey in group:
                for b,bloc,bkey in group:
                    if aloc==bloc:continue
                    if (target['percent'] or target.get('multiplier',False)) and b and (re.search('tokens|total|runs|elements|trials|count|cases|detected|denom',bkey,re.I) or target.get('multiplier',False)):
                        ratio=a/b
                        if matches(target,ratio):return f'({aloc}={a}) / ({bloc}={b})',None
                        if target['percent'] and matches(target,100*ratio):return f'100 × ({aloc}={a}) / ({bloc}={b})',None
                    if re.search('bits|mantissa|width',akey,re.I) and re.search('bits|mantissa|width',bkey,re.I) and matches(target,a-b):
                        return f'({aloc}={a}) - ({bloc}={b})',None
    return None,'not found or derivable by the documented arithmetic rules'


def audit(repo,documents):
    rows=[];hashes={}
    for doc in documents:
        text=repo.read(doc);hashes[doc]=hashlib.sha256(text.encode()).hexdigest();fenced=False
        for lineno,line in enumerate(text.splitlines(),1):
            if re.match(r'\s*(```|~~~)',line):fenced=not fenced;continue
            citations=CITATION.findall(line)
            for target in tokens(line):
                target['context']=line
                row=dict(document=doc,line=lineno,column=target['column'],number=target['raw'],citations=citations)
                if fenced or target['kind']!='number':
                    row.update(status='context',reason='code/date/reference identifier, not a measured claim',proof=None)
                else:
                    proof,reason=prove(target,repo,citations)
                    row.update(status='matched' if proof else 'unmatched',reason=reason,proof=proof)
                rows.append(row)
    return dict(view='staged index' if repo.staged else 'working tree',document_sha256=hashes,
                limitation='Lexical/arithmetic traceability only, not semantic verification; no citation inheritance across lines.',source_sha256={p:hashlib.sha256(v[0].encode()).hexdigest() for p,v in repo.cache.items()},rows=rows)


def table(rows):
    out=['| Document:line | Unmatched number | Cited findings | Reason |','| --- | --- | --- | --- |']
    for row in rows:
        out.append('| '+ ' | '.join(str(x).replace('|','\\|') for x in [f'{row["document"]}:{row["line"]}',row['number'],', '.join(row['citations']) or '—',row['reason']])+' |')
    return '\n'.join(out)+'\n'


def self_test():
    assert [t['raw'] for t in tokens('7.25% [evidence](findings/layer0/file32.json)')]==['7.25']
    assert [t['raw'] for t in tokens('2-64; fp32; 8th; Qwen3-30B-A3B')]==['2','64','8th']
    assert matches(dict(value=Decimal('7.25'),raw='7.25',approx=False),Decimal('7.254'))
    assert not matches(dict(value=Decimal('0'),raw='0',approx=False),Decimal('.001'))
    values,groups=evidence('{"flips": 75208, "tokens": 76852}','fixture.json')
    assert len(values)==2 and len(groups)==1
    assert numeric('10^6')==Decimal(1000000)
    assert [t['raw'] for t in tokens('250x-10^6x, 2.6-2.9x')]==['250','10^6','2.6','2.9']
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'findings/fixture').mkdir(parents=True)
        (root/'findings/fixture/counts.json').write_text('{"flips":2,"tokens":4}')
        repo=Repository(root)
        target=next(tokens('50%'));target['context']='50%'
        proof,reason=prove(target,repo,['findings/fixture/counts.json'])
        assert proof and '/ (' in proof and reason is None
        assert prove(target,repo,[])[1]=='no findings citation on this line'
    print('PASS: tokenizer, rounding, same-record ratios, no citation inheritance and powers')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('documents',nargs='*',default=['README.md','spec/NUMERICS_CONTRACT.md'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--staged',action='store_true')
    p.add_argument('--require-staged-readme',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return 0
    if a.require_staged_readme:
        changed=subprocess.check_output(['git','diff','--cached','--name-only','--','README.md'],cwd=a.root,text=True).splitlines()
        if 'README.md' not in changed:p.error('README.md has not been staged; refusing a premature final audit')
    repo=Repository(a.root,a.staged)
    try:result=audit(repo,a.documents)
    except (OSError,subprocess.CalledProcessError) as exc:p.error(f'cannot read requested document: {exc}')
    unmatched=[r for r in result['rows'] if r['status']=='unmatched'];result['counts']={s:sum(r['status']==s for r in result['rows']) for s in ('matched','unmatched','context')}
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},indent=2));print(table(unmatched))
    if a.out:
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'trace.json').write_text(json.dumps(result,indent=2)+'\n')
        (a.out/'unmatched.md').write_text(table(unmatched))
    return 1 if unmatched else 0
if __name__=='__main__':raise SystemExit(main())
