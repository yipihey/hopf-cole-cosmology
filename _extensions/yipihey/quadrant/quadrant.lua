-- quadrant shortcodes: provenance links into the Rust API documentation and the test suite, and deep links into the lab.
--
--   {{< src spectra::f2_viscous >}}                 -> api/<crate>/spectra/fn.f2_viscous.html   (subtle superscript ⟨f2_viscous⟩)
--   {{< src Cosmo::hopf_cole >}}                    -> api/<crate>/struct.Cosmo.html#method.hopf_cole
--   {{< src hopfcole::HopfColeResult "result" >}}   -> type page, custom text
--   {{< test viscous_kernel_limits "to 10⁻³" >}}   -> badge "✓ to 10⁻³" linking to the test source (text fragment)
--   {{< lab "m=2&ic=w&v=fk" "Open in the lab" >}}   -> button linking to app/index.html#<hash>
--
-- Configuration (optional, in _quarto.yml):
--   quadrant:
--     crate: hcc_core          # crate name as rustdoc spells it (default: the first directory under api/)
--     api: api                 # directory of the rustdoc output, relative to the project root (default api)
--     tests: core/tests        # directory of the test sources (default core/tests)
--     app: app/index.html      # the lab page (default app/index.html)
--
-- Targets are resolved against the files on disk at render time (api/ built by `cargo doc`), so a misspelled item
-- produces a warning and a link to the crate index instead of a dead link. scripts/check_links.py does the same in CI.

local function offset()
  local off = quarto.project and quarto.project.offset or "."
  if off == "" then off = "." end
  return off
end

local function exists_abs(path)
  local f = io.open(path, "r")
  if f then f:close() return true end
  return false
end
local function exists(path) return exists_abs(path) end

local function split(s, sep)
  local out = {}
  for piece in string.gmatch(s, "([^" .. sep .. "]+)") do out[#out + 1] = piece end
  return out
end

local function is_type(name) return name:match("^%u") ~= nil end

local function list_dir(path)
  local ok, entries = pcall(function() return pandoc.system.list_directory(path) end)
  if ok and entries then return entries end
  return {}
end

-- Project root (file checks run from here: Quarto renders each document with its own directory as cwd).
local function project_root()
  local d = quarto.project and quarto.project.directory or nil
  if d and d ~= "" then return d end
  return "."
end

-- Paths in the configuration are relative to the project root. Quarto rewrites path-like metadata relative to each
-- document (`api` becomes `../api` for a chapter in chapters/), so leading ./ and ../ segments are stripped.
local function canon(path)
  path = path:gsub("^%./", "")
  while path:sub(1, 3) == "../" do path = path:sub(4) end
  return (path:gsub("/$", ""))
end

local function conf(meta, key, default)
  local q = meta and meta["quadrant"]
  if q and q[key] then return canon(pandoc.utils.stringify(q[key])) end
  return default
end

local function abs(rel) return project_root() .. "/" .. rel end

local function crate_name(meta)
  local c = conf(meta, "crate", nil)
  if c then return c end
  local api = conf(meta, "api", "api")
  for _, e in ipairs(list_dir(abs(api))) do
    if e ~= "static.files" and e ~= "src" and exists(abs(api .. "/" .. e .. "/index.html")) then return e end
  end
  return "core"
end

local function dep()
  quarto.doc.add_html_dependency({ name = "quadrant", version = "0.1.0", stylesheets = { "quadrant.css" } })
end

-- Resolve a Rust path to a rustdoc page (relative to the project root).
local function resolve_src(root, path)
  local exists = function(rel) return exists_abs(abs(rel)) end
  local parts = split(path, ":")
  local n = #parts
  if n == 0 then return nil end
  local kinds = { "struct", "enum", "trait", "type", "union" }
  local item = parts[n]
  if is_type(item) then
    local dir = root
    for i = 1, n - 1 do dir = dir .. "/" .. parts[i] end
    for _, k in ipairs(kinds) do
      local f = dir .. "/" .. k .. "." .. item .. ".html"
      if exists(f) then return f, "type" end
    end
    return nil
  elseif n >= 2 and is_type(parts[n - 1]) then
    local dir = root
    for i = 1, n - 2 do dir = dir .. "/" .. parts[i] end
    for _, k in ipairs(kinds) do
      local f = dir .. "/" .. k .. "." .. parts[n - 1] .. ".html"
      if exists(f) then return f .. "#method." .. item, "method" end
    end
    return nil
  else
    local dir = root
    for i = 1, n - 1 do dir = dir .. "/" .. parts[i] end
    local f = dir .. "/fn." .. item .. ".html"
    if exists(f) then return f, "fn" end
    if exists(dir .. "/" .. item .. "/index.html") then return dir .. "/" .. item .. "/index.html", "mod" end
    return nil
  end
end

local function src(args, kwargs, meta)
  local path = pandoc.utils.stringify(args[1] or "")
  local text = args[2] and pandoc.utils.stringify(args[2]) or nil
  if path == "" then return pandoc.Null() end
  local parts = split(path, ":")
  local short = text or parts[#parts]
  if not quarto.doc.is_format("html") then return pandoc.Code(path) end
  dep()
  local crate = crate_name(meta)
  local root = conf(meta, "api", "api") .. "/" .. crate
  local target = resolve_src(root, path)
  local full = crate .. "::" .. path
  if not target then
    quarto.log.warning("quadrant src: no rustdoc page for " .. full .. " (build the API documentation first)")
    target = root .. "/index.html"
  end
  local href = offset() .. "/" .. target
  local title = full .. " — implementation (Rust API documentation)"
  return pandoc.RawInline("html", '<sup class="qd-src"><a href="' .. href .. '" title="' .. title .. '">' .. short .. '</a></sup>')
end

-- Find which test file defines `fn <name>(`.
local function find_test(dir, name)
  for _, e in ipairs(list_dir(abs(dir))) do
    if e:match("%.rs$") then
      local h = io.open(abs(dir .. "/" .. e), "r")
      if h then
        local body = h:read("*a"); h:close()
        if body:find("fn%s+" .. name .. "%s*%(") then return dir .. "/" .. e end
      end
    end
  end
  return nil
end

local function repo_url(meta)
  local r = meta["repo-url"] and pandoc.utils.stringify(meta["repo-url"]) or ""
  if r == "" and meta.book and meta.book["repo-url"] then r = pandoc.utils.stringify(meta.book["repo-url"]) end
  if r == "" and meta.website and meta.website["repo-url"] then r = pandoc.utils.stringify(meta.website["repo-url"]) end
  return (r:gsub("/$", ""))
end

local function test(args, kwargs, meta)
  local name = pandoc.utils.stringify(args[1] or "")
  local label = args[2] and pandoc.utils.stringify(args[2]) or "tested"
  if name == "" then return pandoc.Null() end
  if not quarto.doc.is_format("html") then return pandoc.Code(name) end
  dep()
  local dir = conf(meta, "tests", "core/tests")
  local file = find_test(dir, name)
  if not file then quarto.log.warning("quadrant test: no test named " .. name .. " under " .. dir); file = dir end
  local repo = repo_url(meta)
  local branch = meta["repo-branch"] and pandoc.utils.stringify(meta["repo-branch"]) or (meta.book and meta.book["repo-branch"] and pandoc.utils.stringify(meta.book["repo-branch"])) or "main"
  -- repo-subdir (Quarto's own key for projects that live in a subdirectory of the repository)
  local sub = meta["repo-subdir"] and pandoc.utils.stringify(meta["repo-subdir"]) or (meta.book and meta.book["repo-subdir"] and pandoc.utils.stringify(meta.book["repo-subdir"])) or ""
  sub = sub:gsub("^/", ""):gsub("/$", "")
  if sub ~= "" then file = sub .. "/" .. file end
  local href = repo .. "/blob/" .. branch .. "/" .. file .. "#:~:text=fn%20" .. name
  local title = "Verified by the test `" .. name .. "` in " .. file
  return pandoc.RawInline("html", '<a class="qd-test" href="' .. href .. '" title="' .. title .. '">' .. label .. '</a>')
end

local function lab(args, kwargs, meta)
  local hash = pandoc.utils.stringify(args[1] or "")
  local text = args[2] and pandoc.utils.stringify(args[2]) or "Open in the lab"
  if not quarto.doc.is_format("html") then return pandoc.Str(text) end
  dep()
  local page = conf(meta, "app", "app/index.html")
  local href = offset() .. "/" .. page .. (hash ~= "" and ("#" .. hash) or "")
  return pandoc.RawInline("html", '<a class="qd-lab" href="' .. href .. '" title="Opens the laboratory with this experiment set up (the link encodes the whole state)">' .. text .. '</a>')
end

return { ["src"] = src, ["test"] = test, ["lab"] = lab }
