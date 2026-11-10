const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'demo/static/app.js'), 'utf8');
const katex = require(path.join(root, 'demo/static/katex/katex.min.js'));
const functions = source.slice(source.indexOf('function escapeHtml(s)'), source.indexOf('function showError('));
const { renderMarkdown, sanitizeLatex } = vm.runInNewContext(functions + '\n({renderMarkdown, sanitizeLatex})', { katex });
const visible = html => html.replace(/<[^>]*>/g, '');

for (const [name, tex] of [
  ['reported set', String.raw`\square \in \{\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{U}, \mathbf{W}\}`],
  ['literal braces', String.raw`\{a,b\}`],
  ['sized braces', String.raw`\left\{a,b\right\}`],
  ['nested set', String.raw`\{x \mid x \in \{1,2\}\}`],
  ['valid top literal', String.raw`\{top\}`],
  ['valid bot literal', String.raw`\{bot\}`],
]) {
  test(name + ' survives inline and display preview', () => {
    assert.equal(sanitizeLatex(tex), tex);
    const expected = visible(katex.renderToString(tex, { output: 'html', throwOnError: true }));
    for (const markdown of ['$' + tex + '$', '$$' + tex + '$$', '\\[' + tex + '\\]']) {
      const html = renderMarkdown(markdown);
      assert.ok(!html.includes('katex-error'), html);
      assert.ok(visible(html).includes(expected), visible(html));
      assert.equal((visible(html).match(/[{}]/g) || []).length, (expected.match(/[{}]/g) || []).length);
    }
  });
}

test('grouping and superscript braces retain their LaTeX meaning', () => {
  const tex = String.raw`x^{2}+\frac{a+b}{c}`;
  const html = renderMarkdown('$' + tex + '$');
  assert.ok(!html.includes('katex-error'));
  assert.equal((visible(html).match(/[{}]/g) || []).length, 0);
  assert.ok(visible(html).includes('2'));
});

test('literal braces in fenced source remain visible', () => {
  assert.equal(renderMarkdown('```latex\n\\{a,b\\}\n```'), '<pre><code>\\{a,b\\}</code></pre>');
});

test('existing detached equation number normalization still works', () => {
  const html = renderMarkdown('\\[x=1\\]\n\n(5)\n');
  assert.ok(!html.includes('katex-error'));
  assert.ok(visible(html).includes('(5)'));
});
