-- hcc shortcodes: provenance links into the Rust API documentation and the test suite, and deep links into the lab.
--
--   {{< src spectra::f2_viscous >}}                 -> link to api/hcc_core/spectra/fn.f2_viscous.html
--   {{< src Cosmo::hopf_cole >}}                    -> api/hcc_core/struct.Cosmo.html#method.hopf_cole
--   {{< src hopfcole::HopfColeResult "result" >}}   -> type page, custom text
--   {{< test viscous_kernel_limits_and_two_plane_waves >}}       -> badge linking to the test source (text fragment)
--   {{< test name "to 10⁻³" >}}                     -> badge with a label
--   {{< lab "m=2&ic=w&v=fk" "Open in the lab" >}}   -> button linking to app/index.html#<hash>
--
-- Targets are resolved against the files on disk at render time (api/ built by `make api`, core/tests/*.rs), so a
-- misspelled item produces a warning and a link to the module index instead of a dead link.

local CRATE = "hcc_core"

local function offset()
  local off = quarto.project and quarto.project.offset or "."
  if off == "" then off = "." end
  return off
end

local function exists(path)
  local f = io.open(path, "r")
  if f then f:close() return true end
  return false
end

local function split(s, sep)
  local out = {}
  for piece in string.gmatch(s, "([^" .. sep .. "]+)") do out[#out + 1] = piece end
  return out
end

local function is_type(name) return name:match("^%u") ~= nil end

local function dep()
  quarto.doc.add_html_dependency({ name = "hcc-ext", version = "0.1.0", stylesheets = { "hcc-ext.css" } })
end

-- Resolve a Rust path to a rustdoc page (relative to the project root).
local function resolve_src(path)
  local parts = split(path, ":")
  local root = "api/" .. CRATE
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
    -- a module?
    if exists(dir .. "/" .. item .. "/index.html") then return dir .. "/" .. item .. "/index.html", "mod" end
    return nil
  end
end

local function src(args, kwargs)
  local path = pandoc.utils.stringify(args[1] or "")
  local text = args[2] and pandoc.utils.stringify(args[2]) or nil
  if path == "" then return pandoc.Null() end
  local parts = split(path, ":")
  local short = text or parts[#parts]
  if not quarto.doc.is_format("html") then return pandoc.Code(path) end
  dep()
  local target, kind = resolve_src(path)
  local full = CRATE .. "::" .. path
  if not target then
    quarto.log.warning("hcc src: no rustdoc page for " .. full .. " (run `make api`)")
    target = "api/" .. CRATE .. "/index.html"
  end
  local href = offset() .. "/" .. target
  local title = full .. " — implementation (Rust API documentation)"
  return pandoc.RawInline("html", '<sup class="hcc-src"><a href="' .. href .. '" title="' .. title .. '">' .. short .. '</a></sup>')
end

-- Find which test file defines `fn <name>(`.
local function find_test(name)
  local files = { "core/tests/validate.rs", "core/tests/direct.rs", "core/tests/r3d_basic.rs", "core/tests/r3d_differential.rs" }
  for _, f in ipairs(files) do
    local h = io.open(f, "r")
    if h then
      local body = h:read("*a"); h:close()
      if body:find("fn%s+" .. name .. "%s*%(") then return f end
    end
  end
  return nil
end

local function test(args, kwargs, meta)
  local name = pandoc.utils.stringify(args[1] or "")
  local label = args[2] and pandoc.utils.stringify(args[2]) or "tested"
  if name == "" then return pandoc.Null() end
  if not quarto.doc.is_format("html") then return pandoc.Code(name) end
  dep()
  local file = find_test(name)
  if not file then quarto.log.warning("hcc test: no test named " .. name); file = "core/tests/validate.rs" end
  local repo = meta["repo-url"] and pandoc.utils.stringify(meta["repo-url"]) or ""
  if repo == "" and meta.book and meta.book["repo-url"] then repo = pandoc.utils.stringify(meta.book["repo-url"]) end
  repo = repo:gsub("/$", "")
  local href = repo .. "/blob/main/" .. file .. "#:~:text=fn%20" .. name
  local title = "Verified by the test `" .. name .. "` in " .. file .. " (cargo test --release)"
  return pandoc.RawInline("html", '<a class="hcc-test" href="' .. href .. '" title="' .. title .. '">' .. label .. '</a>')
end

local function lab(args, kwargs)
  local hash = pandoc.utils.stringify(args[1] or "")
  local text = args[2] and pandoc.utils.stringify(args[2]) or "Open in the lab"
  if not quarto.doc.is_format("html") then return pandoc.Str(text) end
  dep()
  local href = offset() .. "/app/index.html" .. (hash ~= "" and ("#" .. hash) or "")
  return pandoc.RawInline("html", '<a class="hcc-lab" href="' .. href .. '" title="Opens the laboratory with this experiment set up (the link encodes the whole state)">' .. text .. '</a>')
end

return { ["src"] = src, ["test"] = test, ["lab"] = lab }
