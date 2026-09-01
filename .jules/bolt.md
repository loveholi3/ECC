# Performance optimization in dashboard-web.js

Replaced repetitive and expensive full-document `querySelectorAll` queries in `onSearchInput` with localized DOM queries via parent element caching.

**What:**
Modified lines 717-718 in `scripts/dashboard-web.js`:
* Old code:
```javascript
  document.querySelectorAll('#af .active, #sf .active, #cf .active').forEach(b=>b.classList.remove('active'));
  ['#af button','#sf button','#cf button'].forEach(s=>{const b=document.querySelector(s);if(b)b.classList.add('active')});
```
* New code iteratively fetches the parent containers (`getElementById`) and manipulates DOM using optimal child loop (`getElementsByClassName('active')`) and robust first button assignment.

**Why:**
The previous code invoked full DOM traversals on every keystroke in the search field.

**Measured Improvement:**
* Baseline: ~1209 ms (for 10000 iterations in JSDOM)
* Optimized: ~474 ms
* Over 2.5x performance improvement (~60% execution time reduction).
