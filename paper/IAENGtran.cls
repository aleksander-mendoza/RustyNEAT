%%
%% IAENGtran.cls 2010/09/01
%% 
%% This is the official IAENG LaTeX class for authors of the IAENG journals and
%% conferences. It is based on the version from these support sites:
%% http://www.michaelshell.org/tex/IEEEtran/
%% http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/
%% and
%% Based on the original 1993 IEEEtran.cls, but with many bug fixes
%% and enhancements (from both JVH and MDS) over the 1996/7 version.
%% Contributors:
%% Gerry Murray (1993), Silvano Balemi (1993),
%% Jon Dixon (1996), Peter N"uchter (1996),
%% Juergen von Hagen (2000), and Michael Shell (2001-2007)
%% Copyright (c) 1993-2000 by Gerry Murray, Silvano Balemi, 
%%                         Jon Dixon, Peter N"uchter,
%%                         Juergen von Hagen
%%                         and
%% Copyright (c) 2001-2007 by Michael Shell
%% Current maintainer (V1.3 to V1.7): Michael Shell
%%                                    See:
%%                                    http://www.michaelshell.org/
%%                                    for current contact information.
%% Special thanks to Peter Wilson (CUA) and Donald Arseneau
%% for allowing the inclusion of the \@ifmtarg command 
%% from their ifmtarg LaTeX package. 
%% 
%%*************************************************************************
%% Legal Notice:
%% This code is offered as-is without any warranty either expressed or
%% implied; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE! 
%% User assumes all risk.
%% In no event shall IAENG or any contributor to this code be liable for
%% any damages or losses, including, but not limited to, incidental,
%% consequential, or any other damages, resulting from the use or misuse
%% of any information contained here.
%%
%% All comments are the opinions of their respective authors and are not
%% necessarily endorsed by the IAENG.
%%
%% This work is distributed under the LaTeX Project Public License (LPPL)
%% ( http://www.latex-project.org/ ) version 1.3, and may be freely used,
%% distributed and modified. A copy of the LPPL, version 1.3, is included
%% in the base LaTeX documentation of all distributions of LaTeX released
%% 2003/12/01 or later.
%% Retain all contribution notices and credits.
%% ** Modified files should be clearly indicated as such, including  **
%% ** renaming them and changing author support contact information. **
%% 
%% Major changes to the user interface should be indicated by an 
%% increase in the version numbers. If a version is a beta, it will 
%% be indicated with a BETA suffix, i.e., 1.4 BETA.
%% Small changes can be indicated by appending letters to the version
%% such as "IAENGtran_v14a.cls".
%% In all cases, \Providesclass, any \typeout messages to the user,
%% \IAENGtransversionmajor and \IAENGtransversionminor must reflect the
%% correct version information.
%% The changes should also be documented via source comments.
%%*************************************************************************
%%
%
% Available class options 
% e.g., \documentclass[10pt,conference]{IAENGtran} 
% 
%             *** choose only one from each category ***
%
% 9pt, 10pt, 11pt, 12pt
%    Sets normal font size. The default is 10pt.
% 
% conference, journal, technote, peerreview, peerreviewca
%    determines format mode - conference papers, journal papers,
%    correspondence papers (technotes), or peer review papers. The user
%    should also select 9pt when using technote. peerreview is like
%    journal mode, but provides for a single-column "cover" title page for
%    anonymous peer review. The paper title (without the author names) is
%    repeated at the top of the page after the cover page. For peer review
%    papers, the \IAENGpeerreviewmaketitle command must be executed (will
%    automatically be ignored for non-peerreview modes) at the place the
%    cover page is to end, usually just after the abstract (keywords are
%    not normally used with peer review papers). peerreviewca is like
%    peerreview, but allows the author names to be entered and formatted
%    as with conference mode so that author affiliation and contact
%    information can be easily seen on the cover page.
%    The default is journal.
%
% draft, draftcls, draftclsnofoot, final
%    determines if paper is formatted as a widely spaced draft (for
%    handwritten editor comments) or as a properly typeset final version.
%    draftcls restricts draft mode to the class file while all other LaTeX
%    packages (i.e., \usepackage{graphicx}) will behave as final - allows
%    for a draft paper with visible figures, etc. draftclsnofoot is like
%    draftcls, but does not display the date and the word "DRAFT" at the foot
%    of the pages. If using one of the draft modes, the user will probably
%    also want to select onecolumn.
%    The default is final.
%
% letterpaper, a4paper
%    determines paper size: 8.5in X 11in or 210mm X 297mm. CHANGING THE PAPER
%    SIZE WILL NOT ALTER THE TYPESETTING OF THE DOCUMENT - ONLY THE MARGINS
%    WILL BE AFFECTED. In particular, documents using the a4paper option will
%    have reduced side margins (A4 is narrower than US letter) and a longer
%    bottom margin (A4 is longer than US letter). For both cases, the top
%    margins will be the same and the text will be horizontally centered. 
%    For final submission to IAENG, authors should use A4
%    paper. Note that authors should ensure that all post-processing 
%    (ps, pdf, etc.) uses the same paper specificiation as the .tex document.
%    Problems here are by far the number one reason for incorrect margins.
%    IAENGtran will automatically set the default paper size under pdflatex 
%    (without requiring a change to pdftex.cfg), so this issue is more
%    important to dvips users. Fix config.ps, config.pdf, or ~/.dvipsrc for
%    dvips, or use the dvips -t papersize option instead as needed. See the
%    testflow documentation
%    http://www.ctan.org/tex-archive/macros/latex/contrib/IAENGtran/testflow
%    for more details on dvips paper size configuration.
%    The default is letterpaper.
%
% oneside, twoside
%    determines if layout follows single sided or two sided (duplex)
%    printing. The only notable change is with the headings at the top of
%    the pages.
%    The default is oneside.
%
% onecolumn, twocolumn
%    determines if text is organized into one or two columns per page. One
%    column mode is usually used only with draft papers.
%    The default is twocolumn.
%
% compsoc
%    Use the format of the other IAENG publication.
%
% romanappendices
%    Use the "Appendix I" convention when numbering appendices. IAENGtran.cls
%    now defaults to Alpha "Appendix A" convention - the opposite of what
%    v1.6b and earlier did.
%
% captionsoff
%    disables the display of the figure/table captions. Some IAENG journals
%    request that captions be removed and figures/tables be put on pages
%    of their own at the end of an initial paper submission. The endfloat
%    package can be used with this class option to achieve this format.
%
% nofonttune
%    turns off tuning of the font interword spacing. Maybe useful to those
%    not using the standard Times fonts or for those who have already "tuned"
%    their fonts.
%    The default is to enable IAENGtran to tune font parameters.
%
%
%----------
% Available CLASSINPUTs provided (all are macros unless otherwise noted):
% \CLASSINPUTbaselinestretch
% \CLASSINPUTinnersidemargin
% \CLASSINPUToutersidemargin
% \CLASSINPUTtoptextmargin
% \CLASSINPUTbottomtextmargin
%
% Available CLASSINFOs provided:
% \ifCLASSINFOpdf                       (TeX if conditional)
% \CLASSINFOpaperwidth                  (macro)
% \CLASSINFOpaperheight                 (macro)
% \CLASSINFOnormalsizebaselineskip      (length)
% \CLASSINFOnormalsizeunitybaselineskip (length)
%
% Available CLASSOPTIONs provided:
% all class option flags (TeX if conditionals) unless otherwise noted,
% e.g., \ifCLASSOPTIONcaptionsoff
% point size options provided as a single macro:
% \CLASSOPTIONpt
% which will be defined as 9, 10, 11, or 12 depending on the document's
% normalsize point size.
% also, class option peerreviewca implies the use of class option peerreview
% and classoption draft implies the use of class option draftcls





\ProvidesClass{IAENGtran}[2007/03/05 V1.7a by Michael Shell]
\typeout{-- See the "IAENGtran_HOWTO" manual for usage information.}
\typeout{-- http://www.michaelshell.org/tex/IAENGtran/}
\NeedsTeXFormat{LaTeX2e}

% IAENGtran.cls version numbers, provided as of V1.3
% These values serve as a way a .tex file can
% determine if the new features are provided.
% The version number of this IAENGtrans.cls can be obtained from 
% these values. i.e., V1.4
% KEEP THESE AS INTEGERS! i.e., NO {4a} or anything like that-
% (no need to enumerate "a" minor changes here)
\def\IAENGtransversionmajor{1}
\def\IAENGtransversionminor{7}

% These do nothing, but provide them like in article.cls
\newif\if@restonecol
\newif\if@titlepage


% class option conditionals
\newif\ifCLASSOPTIONonecolumn       \CLASSOPTIONonecolumnfalse
\newif\ifCLASSOPTIONtwocolumn       \CLASSOPTIONtwocolumntrue

\newif\ifCLASSOPTIONoneside         \CLASSOPTIONonesidetrue
\newif\ifCLASSOPTIONtwoside         \CLASSOPTIONtwosidefalse

\newif\ifCLASSOPTIONfinal           \CLASSOPTIONfinaltrue
\newif\ifCLASSOPTIONdraft           \CLASSOPTIONdraftfalse
\newif\ifCLASSOPTIONdraftcls        \CLASSOPTIONdraftclsfalse
\newif\ifCLASSOPTIONdraftclsnofoot  \CLASSOPTIONdraftclsnofootfalse

\newif\ifCLASSOPTIONpeerreview      \CLASSOPTIONpeerreviewfalse
\newif\ifCLASSOPTIONpeerreviewca    \CLASSOPTIONpeerreviewcafalse

\newif\ifCLASSOPTIONjournal         \CLASSOPTIONjournaltrue
\newif\ifCLASSOPTIONconference      \CLASSOPTIONconferencefalse
\newif\ifCLASSOPTIONtechnote        \CLASSOPTIONtechnotefalse

\newif\ifCLASSOPTIONnofonttune      \CLASSOPTIONnofonttunefalse

\newif\ifCLASSOPTIONcaptionsoff     \CLASSOPTIONcaptionsofffalse

\newif\ifCLASSOPTIONcompsoc         \CLASSOPTIONcompsocfalse

\newif\ifCLASSOPTIONromanappendices \CLASSOPTIONromanappendicesfalse


% class info conditionals

% indicates if pdf (via pdflatex) output
\newif\ifCLASSINFOpdf               \CLASSINFOpdffalse


% V1.6b internal flag to show if using a4paper
\newif\if@IAENGusingAfourpaper       \@IAENGusingAfourpaperfalse



% IAENGtran class scratch pad registers
% dimen
\newdimen\@IAENGtrantmpdimenA
\newdimen\@IAENGtrantmpdimenB
% count
\newcount\@IAENGtrantmpcountA
\newcount\@IAENGtrantmpcountB
% token list
\newtoks\@IAENGtrantmptoksA

% we use \CLASSOPTIONpt so that we can ID the point size (even for 9pt docs)
% as well as LaTeX's \@ptsize to retain some compatability with some
% external packages
\def\@ptsize{0}
% LaTeX does not support 9pt, so we set \@ptsize to 0 - same as that of 10pt
\DeclareOption{9pt}{\def\CLASSOPTIONpt{9}\def\@ptsize{0}}
\DeclareOption{10pt}{\def\CLASSOPTIONpt{10}\def\@ptsize{0}}
\DeclareOption{11pt}{\def\CLASSOPTIONpt{11}\def\@ptsize{1}}
\DeclareOption{12pt}{\def\CLASSOPTIONpt{12}\def\@ptsize{2}}



\DeclareOption{letterpaper}{\setlength{\paperheight}{11in}%
                            \setlength{\paperwidth}{8.5in}%
                            \@IAENGusingAfourpaperfalse
                            \def\CLASSOPTIONpaper{letter}%
                            \def\CLASSINFOpaperwidth{8.5in}%
                            \def\CLASSINFOpaperheight{11in}}


\DeclareOption{a4paper}{\setlength{\paperheight}{297mm}%
                        \setlength{\paperwidth}{210mm}%
                        \@IAENGusingAfourpapertrue
                        \def\CLASSOPTIONpaper{a4}%
                        \def\CLASSINFOpaperwidth{210mm}%
                        \def\CLASSINFOpaperheight{297mm}}

\DeclareOption{oneside}{\@twosidefalse\@mparswitchfalse
                        \CLASSOPTIONonesidetrue\CLASSOPTIONtwosidefalse}
\DeclareOption{twoside}{\@twosidetrue\@mparswitchtrue
                        \CLASSOPTIONtwosidetrue\CLASSOPTIONonesidefalse}

\DeclareOption{onecolumn}{\CLASSOPTIONonecolumntrue\CLASSOPTIONtwocolumnfalse}
\DeclareOption{twocolumn}{\CLASSOPTIONtwocolumntrue\CLASSOPTIONonecolumnfalse}

% If the user selects draft, then this class AND any packages
% will go into draft mode.
\DeclareOption{draft}{\CLASSOPTIONdrafttrue\CLASSOPTIONdraftclstrue
                      \CLASSOPTIONdraftclsnofootfalse} 
% draftcls is for a draft mode which will not affect any packages
% used by the document.
\DeclareOption{draftcls}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
                         \CLASSOPTIONdraftclsnofootfalse} 
% draftclsnofoot is like draftcls, but without the footer.
\DeclareOption{draftclsnofoot}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
                               \CLASSOPTIONdraftclsnofoottrue} 
\DeclareOption{final}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclsfalse
                      \CLASSOPTIONdraftclsnofootfalse}

\DeclareOption{journal}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                        \CLASSOPTIONjournaltrue\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{conference}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                           \CLASSOPTIONjournalfalse\CLASSOPTIONconferencetrue\CLASSOPTIONtechnotefalse}

\DeclareOption{technote}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                         \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotetrue}

\DeclareOption{peerreview}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcafalse
                           \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{peerreviewca}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcatrue
                             \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{nofonttune}{\CLASSOPTIONnofonttunetrue}

\DeclareOption{captionsoff}{\CLASSOPTIONcaptionsofftrue}

\DeclareOption{compsoc}{\CLASSOPTIONcompsoctrue}

\DeclareOption{romanappendices}{\CLASSOPTIONromanappendicestrue}


% default to US letter paper, 10pt, twocolumn, one sided, final, journal
\ExecuteOptions{a4paper,10pt,twocolumn,oneside,final,journal}
% overrride these defaults per user requests
\ProcessOptions



% Computer Society conditional execution command
\long\def\@IAENGcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\relax#1\relax\fi\relax}
% inverse
\long\def\@IAENGnotcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\else\relax#1\relax\fi\relax}
% compsoc conference
\long\def\@IAENGcompsocconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\relax#1\relax\fi\fi\relax}
% compsoc not conference
\long\def\@IAENGcompsocnotconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\else\relax#1\relax\fi\fi\relax}


% IAENG uses Times Roman font, so we'll default to Times.
% These three commands make up the entire times.sty package.
\renewcommand{\sfdefault}{phv}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\ttdefault}{pcr}

\@IAENGcompsoconly{\typeout{-- Using IAENG Computer Society mode.}}

% V1.7 compsoc nonconference papers, use Palatino/Palladio as the main text font,
% not Times Roman.
\@IAENGcompsocnotconfonly{\renewcommand{\rmdefault}{ppl}}

% enable Times/Palatino main text font
\normalfont\selectfont





% V1.7 conference notice message hook
\def\@IAENGconsolenoticeconference{\typeout{}%
\typeout{** Conference Paper **}%
\typeout{Before submitting the final camera ready copy, remember to:}%
\typeout{}%
\typeout{ 1. Manually equalize the lengths of two columns on the last page}%
\typeout{ of your paper;}%
\typeout{}%
\typeout{ 2. Ensure that any PostScript and/or PDF output post-processing}%
\typeout{ uses only Type 1 fonts and that every step in the generation}%
\typeout{ process uses the appropriate paper size.}%
\typeout{}}


% we can send console reminder messages to the user here
\AtEndDocument{\ifCLASSOPTIONconference\@IAENGconsolenoticeconference\fi}


% warn about the use of single column other than for draft mode
\ifCLASSOPTIONtwocolumn\else%
  \ifCLASSOPTIONdraftcls\else%
   \typeout{** ATTENTION: Single column mode is not typically used with IAENG publications.}%
  \fi%
\fi


% V1.7 improved paper size setting code.
% Set pdfpage and dvips paper sizes. Conditional tests are similar to that
% of ifpdf.sty. Retain within {} to ensure tested macros are never altered,
% even if only effect is to set them to \relax.
% if \pdfoutput is undefined or equal to relax, output a dvips special
{\@ifundefined{pdfoutput}{\AtBeginDvi{\special{papersize=\CLASSINFOpaperwidth,\CLASSINFOpaperheight}}}{%
% pdfoutput is defined and not equal to \relax
% check for pdfpageheight existence just in case someone sets pdfoutput
% under non-pdflatex. If exists, set them regardless of value of \pdfoutput.
\@ifundefined{pdfpageheight}{\relax}{\global\pdfpagewidth\paperwidth
\global\pdfpageheight\paperheight}%
% if using \pdfoutput=0 under pdflatex, send dvips papersize special
\ifcase\pdfoutput
\AtBeginDvi{\special{papersize=\CLASSINFOpaperwidth,\CLASSINFOpaperheight}}%
\else
% we are using pdf output, set CLASSINFOpdf flag
\global\CLASSINFOpdftrue
\fi}}

% let the user know the selected papersize
\typeout{-- Using \CLASSINFOpaperwidth\space x \CLASSINFOpaperheight\space
(\CLASSOPTIONpaper)\space paper.}

\ifCLASSINFOpdf
\typeout{-- Using PDF output.}
\else
\typeout{-- Using DVI output.}
\fi


% The idea hinted here is for LaTeX to generate markleft{} and markright{}
% automatically for you after you enter \author{}, \journal{},
% \journaldate{}, journalvol{}, \journalnum{}, etc.
% However, there may be some backward compatibility issues here as
% well as some special applications for IAENGtran.cls and special issues
% that may require the flexible \markleft{}, \markright{} and/or \markboth{}.
% We'll leave this as an open future suggestion.
%\newcommand{\journal}[1]{\def\@journal{#1}}
%\def\@journal{}



% pointsize values
% used with ifx to determine the document's normal size
\def\@IAENGptsizenine{9}
\def\@IAENGptsizeten{10}
\def\@IAENGptsizeeleven{11}
\def\@IAENGptsizetwelve{12}



% FONT DEFINITIONS (No sizexx.clo file needed) 
% V1.6 revised font sizes, displayskip values and
%      revised normalsize baselineskip to reduce underfull vbox problems
%      on the 58pc = 696pt = 9.5in text height we want
%      normalsize     #lines/column  baselineskip (aka leading)
%             9pt     63             11.0476pt (truncated down)
%            10pt     58             12pt      (exact)
%            11pt     52             13.3846pt (truncated down)
%            12pt     50             13.92pt   (exact)
%

% we need to store the nominal baselineskip for the given font size
% in case baselinestretch ever changes.
% this is a dimen, so it will not hold stretch or shrink
\newdimen\@IAENGnormalsizeunitybaselineskip
\@IAENGnormalsizeunitybaselineskip\baselineskip

\ifx\CLASSOPTIONpt\@IAENGptsizenine
\typeout{-- This is a 9 point document.}
\def\normalsize{\@setfontsize{\normalsize}{9}{11.0476pt}}%
\setlength{\@IAENGnormalsizeunitybaselineskip}{11.0476pt}%
\normalsize
\abovedisplayskip 1.5ex plus3pt minus1pt%
\belowdisplayskip \abovedisplayskip%
\abovedisplayshortskip 0pt plus3pt%
\belowdisplayshortskip 1.5ex plus3pt minus1pt
\def\small{\@setfontsize{\small}{8.5}{10pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{8}{9pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{7}{8pt}}
\def\tiny{\@setfontsize{\tiny}{5}{6pt}}
% sublargesize is the same as large - 10pt
\def\sublargesize{\@setfontsize{\sublargesize}{10}{12pt}}
\def\large{\@setfontsize{\large}{10}{12pt}}
\def\Large{\@setfontsize{\Large}{12}{14pt}}
\def\LARGE{\@setfontsize{\LARGE}{14}{17pt}}
\def\huge{\@setfontsize{\huge}{17}{20pt}}
\def\Huge{\@setfontsize{\Huge}{20}{24pt}}
\fi


% Check if we have selected 10 points
\ifx\CLASSOPTIONpt\@IAENGptsizeten
\typeout{-- This is a 10 point document.}
\def\normalsize{\@setfontsize{\normalsize}{10}{12.00pt}}%
\setlength{\@IAENGnormalsizeunitybaselineskip}{12pt}%
\normalsize
\abovedisplayskip 1.5ex plus4pt minus2pt%
\belowdisplayskip \abovedisplayskip%
\abovedisplayshortskip 0pt plus4pt%
\belowdisplayshortskip 1.5ex plus4pt minus2pt
\def\small{\@setfontsize{\small}{9}{10pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{8}{9pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{7}{8pt}}
\def\tiny{\@setfontsize{\tiny}{5}{6pt}}
% sublargesize is a tad smaller than large - 11pt
\def\sublargesize{\@setfontsize{\sublargesize}{11}{13.4pt}}
\def\large{\@setfontsize{\large}{12}{14pt}}
\def\Large{\@setfontsize{\Large}{14}{17pt}}
\def\LARGE{\@setfontsize{\LARGE}{17}{20pt}}
\def\huge{\@setfontsize{\huge}{20}{24pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi


% Check if we have selected 11 points
\ifx\CLASSOPTIONpt\@IAENGptsizeeleven
\typeout{-- This is an 11 point document.}
\def\normalsize{\@setfontsize{\normalsize}{11}{13.3846pt}}%
\setlength{\@IAENGnormalsizeunitybaselineskip}{13.3846pt}%
\normalsize
\abovedisplayskip 1.5ex plus5pt minus3pt%
\belowdisplayskip \abovedisplayskip%
\abovedisplayshortskip 0pt plus5pt%
\belowdisplayshortskip 1.5ex plus5pt minus3pt
\def\small{\@setfontsize{\small}{10}{12pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{9}{10.5pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{8}{9pt}}
\def\tiny{\@setfontsize{\tiny}{6}{7pt}}
% sublargesize is the same as large - 12pt
\def\sublargesize{\@setfontsize{\sublargesize}{12}{14pt}}
\def\large{\@setfontsize{\large}{12}{14pt}}
\def\Large{\@setfontsize{\Large}{14}{17pt}}
\def\LARGE{\@setfontsize{\LARGE}{17}{20pt}}
\def\huge{\@setfontsize{\huge}{20}{24pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi


% Check if we have selected 12 points
\ifx\CLASSOPTIONpt\@IAENGptsizetwelve
\typeout{-- This is a 12 point document.}
\def\normalsize{\@setfontsize{\normalsize}{12}{13.92pt}}%
\setlength{\@IAENGnormalsizeunitybaselineskip}{13.92pt}%
\normalsize
\abovedisplayskip 1.5ex plus6pt minus4pt%
\belowdisplayskip \abovedisplayskip%
\abovedisplayshortskip 0pt plus6pt%
\belowdisplayshortskip 1.5ex plus6pt minus4pt
\def\small{\@setfontsize{\small}{10}{12pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{9}{10.5pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{8}{9pt}}
\def\tiny{\@setfontsize{\tiny}{6}{7pt}}
% sublargesize is the same as large - 14pt
\def\sublargesize{\@setfontsize{\sublargesize}{14}{17pt}}
\def\large{\@setfontsize{\large}{14}{17pt}}
\def\Large{\@setfontsize{\Large}{17}{20pt}}
\def\LARGE{\@setfontsize{\LARGE}{20}{24pt}}
\def\huge{\@setfontsize{\huge}{22}{26pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi


% V1.6 The Computer Modern Fonts will issue a substitution warning for
% 24pt titles (24.88pt is used instead) increase the substitution
% tolerance to turn off this warning
\def\fontsubfuzz{.9pt}
% However, the default (and correct) Times font will scale exactly as needed.


% warn the user in case they forget to use the 9pt option with
% technote
\ifCLASSOPTIONtechnote%
 \ifx\CLASSOPTIONpt\@IAENGptsizenine\else%
  \typeout{** ATTENTION: Technotes are normally 9pt documents.}%
 \fi%
\fi


% V1.7
% Improved \textunderscore to provide a much better fake _ when used with
% OT1 encoding. Under OT1, detect use of pcr or cmtt \ttfamily and use
% available true _ glyph for those two typewriter fonts.
\def\@IAENGstringptm{ptm} % Times Roman family
\def\@IAENGstringppl{ppl} % Palatino Roman family
\def\@IAENGstringphv{phv} % Helvetica Sans Serif family
\def\@IAENGstringpcr{pcr} % Courier typewriter family
\def\@IAENGstringcmtt{cmtt} % Computer Modern typewriter family
\DeclareTextCommandDefault{\textunderscore}{\leavevmode
\ifx\f@family\@IAENGstringpcr\string_\else
\ifx\f@family\@IAENGstringcmtt\string_\else
\ifx\f@family\@IAENGstringptm\kern 0em\vbox{\hrule\@width 0.5em\@height 0.5pt\kern -0.3ex}\else
\ifx\f@family\@IAENGstringppl\kern 0em\vbox{\hrule\@width 0.5em\@height 0.5pt\kern -0.3ex}\else
\ifx\f@family\@IAENGstringphv\kern -0.03em\vbox{\hrule\@width 0.62em\@height 0.52pt\kern -0.33ex}\kern -0.03em\else
\kern 0.09em\vbox{\hrule\@width 0.6em\@height 0.44pt\kern -0.63pt\kern -0.42ex}\kern 0.09em\fi\fi\fi\fi\fi\relax}




% set the default \baselinestretch
\def\baselinestretch{1}
\ifCLASSOPTIONdraftcls
  \def\baselinestretch{1.5}% default baselinestretch for draft modes
\fi 


% process CLASSINPUT baselinestretch
\ifx\CLASSINPUTbaselinestretch\@IAENGundefined
\else
  \edef\baselinestretch{\CLASSINPUTbaselinestretch} % user CLASSINPUT override
  \typeout{** ATTENTION: Overriding \string\baselinestretch\space to
           \baselinestretch\space via \string\CLASSINPUT.}
\fi

\normalsize % make \baselinestretch take affect




% store the normalsize baselineskip
\newdimen\CLASSINFOnormalsizebaselineskip
\CLASSINFOnormalsizebaselineskip=\baselineskip\relax
% and the normalsize unity (baselinestretch=1) baselineskip
% we could save a register by giving the user access to
% \@IAENGnormalsizeunitybaselineskip. However, let's protect
% its read only internal status
\newdimen\CLASSINFOnormalsizeunitybaselineskip
\CLASSINFOnormalsizeunitybaselineskip=\@IAENGnormalsizeunitybaselineskip\relax
% store the nominal value of jot
\newdimen\IAENGnormaljot
\IAENGnormaljot=0.25\baselineskip\relax

% set \jot
\jot=\IAENGnormaljot\relax




% V1.6, we are now going to fine tune the interword spacing
% The default interword glue for Times under TeX appears to use a
% nominal interword spacing of 25% (relative to the font size, i.e., 1em)
% a maximum of 40% and a minimum of 19%.
% For example, 10pt text uses an interword glue of:
% 
% 2.5pt plus 1.49998pt minus 0.59998pt
% 
% However, IAENG allows for a more generous range which reduces the need
% for hyphenation, especially for two column text. Furthermore, IAENG
% tends to use a little bit more nominal space between the words.
% IAENG's interword spacing percentages appear to be:
% 35% nominal
% 23% minimum
% 50% maximum
% (They may even be using a tad more for the largest fonts such as 24pt.)
% 
% for bold text, IAENG increases the spacing a little more:
% 37.5% nominal
% 23% minimum
% 55% maximum

% here are the interword spacing ratios we'll use
% for medium (normal weight)
\def\@IAENGinterspaceratioM{0.35}
\def\@IAENGinterspaceMINratioM{0.23}
\def\@IAENGinterspaceMAXratioM{0.50}

% for bold
\def\@IAENGinterspaceratioB{0.375}
\def\@IAENGinterspaceMINratioB{0.23}
\def\@IAENGinterspaceMAXratioB{0.55}


% command to revise the interword spacing for the current font under TeX:
% \fontdimen2 = nominal interword space
% \fontdimen3 = interword stretch
% \fontdimen4 = interword shrink
% since all changes to the \fontdimen are global, we can enclose these commands
% in braces to confine any font attribute or length changes
\def\@@@IAENGsetfontdimens#1#2#3{{%
\setlength{\@IAENGtrantmpdimenB}{\f@size pt}% grab the font size in pt, could use 1em instead.
\setlength{\@IAENGtrantmpdimenA}{#1\@IAENGtrantmpdimenB}%
\fontdimen2\font=\@IAENGtrantmpdimenA\relax
\addtolength{\@IAENGtrantmpdimenA}{-#2\@IAENGtrantmpdimenB}%
\fontdimen3\font=-\@IAENGtrantmpdimenA\relax
\setlength{\@IAENGtrantmpdimenA}{#1\@IAENGtrantmpdimenB}%
\addtolength{\@IAENGtrantmpdimenA}{-#3\@IAENGtrantmpdimenB}%
\fontdimen4\font=\@IAENGtrantmpdimenA\relax}}

% revise the interword spacing for each font weight
\def\@@IAENGsetfontdimens{{%
\mdseries
\@@@IAENGsetfontdimens{\@IAENGinterspaceratioM}{\@IAENGinterspaceMAXratioM}{\@IAENGinterspaceMINratioM}%
\bfseries
\@@@IAENGsetfontdimens{\@IAENGinterspaceratioB}{\@IAENGinterspaceMAXratioB}{\@IAENGinterspaceMINratioB}%
}}

% revise the interword spacing for each font shape
% \slshape is not often used for IAENG work and is not altered here. The \scshape caps are
% already a tad too large in the free LaTeX fonts (as compared to what IAENG uses) so we
% won't alter these either.
\def\@IAENGsetfontdimens{{%
\normalfont
\@@IAENGsetfontdimens
\normalfont\itshape
\@@IAENGsetfontdimens
}}

% command to revise the interword spacing for each font size (and shape
% and weight). Only the \rmfamily is done here as \ttfamily uses a 
% fixed spacing and \sffamily is not used as the main text of IAENG papers.
\def\@IAENGtunefonts{{\selectfont\rmfamily
\tiny\@IAENGsetfontdimens
\scriptsize\@IAENGsetfontdimens
\footnotesize\@IAENGsetfontdimens
\small\@IAENGsetfontdimens
\normalsize\@IAENGsetfontdimens
\sublargesize\@IAENGsetfontdimens
\large\@IAENGsetfontdimens
\LARGE\@IAENGsetfontdimens
\huge\@IAENGsetfontdimens
\Huge\@IAENGsetfontdimens}}

% if the nofonttune class option is not given, revise the interword spacing
% now - in case IAENGtran makes any default length measurements, and make
% sure all the default fonts are loaded
\ifCLASSOPTIONnofonttune\else
\@IAENGtunefonts
\fi

% and again at the start of the document in case the user loaded different fonts
\AtBeginDocument{\ifCLASSOPTIONnofonttune\else\@IAENGtunefonts\fi}



% V1.6 
% LaTeX is a little to quick to use hyphenations
% So, we increase the penalty for their use and raise
% the badness level that triggers an underfull hbox
% warning. The author may still have to tweak things,
% but the appearance will be much better "right out
% of the box" than that under V1.5 and prior.
% TeX default is 50
\hyphenpenalty=750
% If we didn't adjust the interword spacing, 2200 might be better.
% The TeX default is 1000
\hbadness=1350
% IAENG does not use extra spacing after punctuation
\frenchspacing

% V1.7 increase this a tad to discourage equation breaks
\binoppenalty=1000 % default 700
\relpenalty=800     % default 500


% margin note stuff
\marginparsep      10pt
\marginparwidth    20pt
\marginparpush     25pt


% if things get too close, go ahead and let them touch
\lineskip            0pt
\normallineskip      0pt
\lineskiplimit       0pt
\normallineskiplimit 0pt

% The distance from the lower edge of the text body to the
% footline
\footskip 0.4in

% normally zero, should be relative to font height.
% put in a little rubber to help stop some bad breaks (underfull vboxes)
\parskip 0ex plus 0.2ex minus 0.1ex

\parindent    1.0em

\topmargin    -49.0pt
\headheight   12pt
\headsep      0.25in

% use the normal font baselineskip
% so that \topskip is unaffected by changes in \baselinestretch
\topskip=\@IAENGnormalsizeunitybaselineskip
\textheight     58pc       % 58pc, 9.63in, 696pt
% Tweak textheight to a perfect integer number of lines/page.
% The normal baselineskip for each document point size is used 
% to determine these values.
\ifx\CLASSOPTIONpt\@IAENGptsizenine\textheight=63\@IAENGnormalsizeunitybaselineskip\fi      % 63 lines/page
\ifx\CLASSOPTIONpt\@IAENGptsizeten\textheight=58\@IAENGnormalsizeunitybaselineskip\fi       % 58 lines/page
\ifx\CLASSOPTIONpt\@IAENGptsizeeleven\textheight=52\@IAENGnormalsizeunitybaselineskip\fi    % 52 lines/page
\ifx\CLASSOPTIONpt\@IAENGptsizetwelve\textheight=50\@IAENGnormalsizeunitybaselineskip\fi    % 50 lines/page

\setlength{\textheight}{10.15in}
\columnsep         1pc
\textwidth        42pc   % 2 x 21pc + 1pc = 43pc
 

% the default side margins are equal
\if@IAENGusingAfourpaper 
\oddsidemargin        16.51mm
\evensidemargin       16.51mm
\else
\oddsidemargin        0.680in
\evensidemargin       0.680in
\fi
% compensate for LaTeX's 1in offset
\addtolength{\oddsidemargin}{-1in}
\addtolength{\evensidemargin}{-1in}



% adjust margins for conference mode
\ifCLASSOPTIONconference
 \topmargin        -0.25in
 % we retain the reserved, but unused space for headers
 \addtolength{\topmargin}{-\headheight}
 \addtolength{\topmargin}{-\headsep}
 %\textheight        9.25in % The standard for conferences (668.4975pt)
 % Tweak textheight to a perfect integer number of lines/page.
 \ifx\CLASSOPTIONpt\@IAENGptsizenine\textheight=61\@IAENGnormalsizeunitybaselineskip\fi      % 61 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizeten\textheight=56\@IAENGnormalsizeunitybaselineskip\fi       % 56 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizeeleven\textheight=50\@IAENGnormalsizeunitybaselineskip\fi    % 50 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizetwelve\textheight=48\@IAENGnormalsizeunitybaselineskip\fi    % 48 lines/page
\fi


% compsoc conference
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference
 % compsoc conference use a larger value for columnsep
 \columnsep 0.375in
 % compsoc conferences want 1in top margin, 1.125in bottom margin
 \topmargin        0in
 \addtolength{\topmargin}{-6pt}% we tweak this a tad to better comply with top of line stuff
 % we retain the reserved, but unused space for headers
 \addtolength{\topmargin}{-\headheight}
 \addtolength{\topmargin}{-\headsep}
 \textheight        8.875in % (641.39625pt)
 % Tweak textheight to a perfect integer number of lines/page.
 \ifx\CLASSOPTIONpt\@IAENGptsizenine\textheight=58\@IAENGnormalsizeunitybaselineskip\fi      % 58 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizeten\textheight=53\@IAENGnormalsizeunitybaselineskip\fi       % 53 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizeeleven\textheight=48\@IAENGnormalsizeunitybaselineskip\fi    % 48 lines/page
 \ifx\CLASSOPTIONpt\@IAENGptsizetwelve\textheight=46\@IAENGnormalsizeunitybaselineskip\fi    % 46 lines/page 
 \textwidth 6.5in
 % the default side margins are equal
 \if@IAENGusingAfourpaper 
  \oddsidemargin        22.45mm
  \evensidemargin       22.45mm
 \else
  \oddsidemargin        1in
  \evensidemargin       1in
 \fi
 % compensate for LaTeX's 1in offset
 \addtolength{\oddsidemargin}{-1in}
 \addtolength{\evensidemargin}{-1in}
\fi\fi



% draft mode settings override that of all other modes
% provides a nice 1in margin all around the paper and extra
% space between the lines for editor's comments
\ifCLASSOPTIONdraftcls 
  % want 1in from top of paper to text
  \setlength{\topmargin}{-\headsep}%
  \addtolength{\topmargin}{-\headheight}%
  % we want 1in side margins regardless of paper type
  \oddsidemargin      0in
  \evensidemargin     0in
  % set the text width
  \setlength{\textwidth}{\paperwidth}%
  \addtolength{\textwidth}{-2.0in}%
  \setlength{\textheight}{\paperheight}%
  \addtolength{\textheight}{-2.0in}%
  % digitize textheight to be an integer number of lines.
  % this may cause the bottom margin to be off a tad
  \addtolength{\textheight}{-1\topskip}%
  \divide\textheight  by \baselineskip%
  \multiply\textheight  by \baselineskip%
  \addtolength{\textheight}{\topskip}%
\fi



% process CLASSINPUT inner/outer margin
% if inner margin defined, but outer margin not, set outer to inner.
\ifx\CLASSINPUTinnersidemargin\@IAENGundefined
\else
  \ifx\CLASSINPUToutersidemargin\@IAENGundefined
    \edef\CLASSINPUToutersidemargin{\CLASSINPUTinnersidemargin}
  \fi
\fi

\ifx\CLASSINPUToutersidemargin\@IAENGundefined
\else
  % if outer margin defined, but inner margin not, set inner to outer.
  \ifx\CLASSINPUTinnersidemargin\@IAENGundefined
    \edef\CLASSINPUTinnersidemargin{\CLASSINPUToutersidemargin}
  \fi
  \setlength{\oddsidemargin}{\CLASSINPUTinnersidemargin}
  \ifCLASSOPTIONtwoside
    \setlength{\evensidemargin}{\CLASSINPUToutersidemargin}
  \else
    \setlength{\evensidemargin}{\CLASSINPUTinnersidemargin}
  \fi
  \addtolength{\oddsidemargin}{-1in}
  \addtolength{\evensidemargin}{-1in}
  \setlength{\textwidth}{\paperwidth}
  \addtolength{\textwidth}{-\CLASSINPUTinnersidemargin}
  \addtolength{\textwidth}{-\CLASSINPUToutersidemargin}
  \typeout{** ATTENTION: Overriding inner side margin to \CLASSINPUTinnersidemargin\space and 
           outer side margin to \CLASSINPUToutersidemargin\space via \string\CLASSINPUT.}
\fi



% process CLASSINPUT top/bottom text margin
% if toptext margin defined, but bottomtext margin not, set bottomtext to toptext margin
\ifx\CLASSINPUTtoptextmargin\@IAENGundefined
\else
  \ifx\CLASSINPUTbottomtextmargin\@IAENGundefined
    \edef\CLASSINPUTbottomtextmargin{\CLASSINPUTtoptextmargin}
  \fi
\fi

\ifx\CLASSINPUTbottomtextmargin\@IAENGundefined
\else
  % if bottomtext margin defined, but toptext margin not, set toptext to bottomtext margin
  \ifx\CLASSINPUTtoptextmargin\@IAENGundefined
    \edef\CLASSINPUTtoptextmargin{\CLASSINPUTbottomtextmargin}
  \fi
  \setlength{\topmargin}{\CLASSINPUTtoptextmargin}
  \addtolength{\topmargin}{-1in}
  \addtolength{\topmargin}{-\headheight}
  \addtolength{\topmargin}{-\headsep}
  \setlength{\textheight}{\paperheight}
  \addtolength{\textheight}{-\CLASSINPUTtoptextmargin}
  \addtolength{\textheight}{-\CLASSINPUTbottomtextmargin}
  % in the default format we use the normal baselineskip as topskip
  % we only need 0.7 of this to clear typical top text and we need
  % an extra 0.3 spacing at the bottom for descenders. This will
  % correct for both.
  \addtolength{\topmargin}{-0.3\@IAENGnormalsizeunitybaselineskip}
  \typeout{** ATTENTION: Overriding top text margin to \CLASSINPUTtoptextmargin\space and 
           bottom text margin to \CLASSINPUTbottomtextmargin\space via \string\CLASSINPUT.}
\fi







% LIST SPACING CONTROLS

% Controls the amount of EXTRA spacing
% above and below \trivlist 
% Both \list and IED lists override this.
% However, \trivlist will use this as will most
% things built from \trivlist like the \center
% environment.
\topsep           0.5\baselineskip

% Controls the additional spacing around lists preceded
% or followed by blank lines. IAENG does not increase
% spacing before or after paragraphs so it is set to zero.
% \z@ is the same as zero, but faster.
\partopsep          \z@

% Controls the spacing between paragraphs in lists. 
% IAENG does not increase spacing before or after paragraphs
% so this is also zero. 
% With IAENGtran.cls, global changes to
% this value DO affect lists (but not IED lists).
\parsep             \z@

% Controls the extra spacing between list items. 
% IAENG does not put extra spacing between items.
% With IAENGtran.cls, global changes to this value DO affect
% lists (but not IED lists).
\itemsep            \z@

% \itemindent is the amount to indent the FIRST line of a list
% item. It is auto set to zero within the \list environment. To alter
% it, you have to do so when you call the \list.
% However, IAENG uses this for the theorem environment
% There is an alternative value for this near \leftmargini below
\itemindent         -1em

% \leftmargin, the spacing from the left margin of the main text to
% the left of the main body of a list item is set by \list.
% Hence this statement does nothing for lists.
% But, quote and verse do use it for indention.
\leftmargin         2em

% we retain this stuff from the older IAENGtran.cls so that \list
% will work the same way as before. However, itemize, enumerate and
% description (IED) could care less about what these are as they
% all are overridden.
\leftmargini        2em
%\itemindent         2em  % Alternative values: sometimes used.
%\leftmargini        0em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep         0.5em 
\labelwidth         \z@


% The old IAENGtran.cls behavior of \list is retained.
% However, the new V1.3 IED list environments override all the
% @list stuff (\@listX is called within \list for the
% appropriate level just before the user's list_decl is called). 
% \topsep is now 2pt as IAENG puts a little extra space around
% lists - used by those non-IED macros that depend on \list.
% Note that \parsep and \itemsep are not redefined as in 
% the sizexx.clo \@listX (which article.cls uses) so global changes
% of these values DO affect \list
% 
\def\@listi{\leftmargin\leftmargini \topsep 2pt plus 1pt minus 1pt}
\let\@listI\@listi
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi%
    \advance\labelwidth-\labelsep \topsep 2pt}


% IAENG uses 5) not 5.
\def\labelenumi{\theenumi)}     \def\theenumi{\arabic{enumi}}

% IAENG uses a) not (a)
\def\labelenumii{\theenumii)}  \def\theenumii{\alph{enumii}}

% IAENG uses iii) not iii.
\def\labelenumiii{\theenumiii)} \def\theenumiii{\roman{enumiii}}

% IAENG uses A) not A.
\def\labelenumiv{\theenumiv)}   \def\theenumiv{\Alph{enumiv}}

% exactly the same as in article.cls
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

% itemized list label styles
\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\textbf{--}}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}



% **** V1.3 ENHANCEMENTS ****
% Itemize, Enumerate and Description (IED) List Controls
% ***************************
% 
% 
% IAENG seems to use at least two different values by
% which ITEMIZED list labels are indented to the right
% For The Journal of Lightwave Technology (JLT) and The Journal
% on Selected Areas in Communications (JSAC), they tend to use
% an indention equal to \parindent. For Transactions on Communications
% they tend to indent ITEMIZED lists a little more--- 1.3\parindent.
% We'll provide both values here for you so that you can choose 
% which one you like in your document using a command such as:
% setlength{\IAENGilabelindent}{\IAENGilabelindentB}
\newdimen\IAENGilabelindentA
\IAENGilabelindentA \parindent

\newdimen\IAENGilabelindentB
\IAENGilabelindentB 1.3\parindent
% However, we'll default to using \parindent
% which makes more sense to me
\newdimen\IAENGilabelindent
\IAENGilabelindent \IAENGilabelindentA


% This controls the default amount the enumerated list labels
% are indented to the right.
% Normally, this is the same as the paragraph indention
\newdimen\IAENGelabelindent
\IAENGelabelindent \parindent

% This controls the default amount the description list labels
% are indented to the right.
% Normally, this is the same as the paragraph indention
\newdimen\IAENGdlabelindent
\IAENGdlabelindent \parindent

% This is the value actually used within the IED lists.
% The IED environments automatically set its value to
% one of the three values above, so global changes do 
% not have any effect
\newdimen\IAENGlabelindent
\IAENGlabelindent \parindent

% The actual amount labels will be indented is
% \IAENGlabelindent multiplied by the factor below
% corresponding to the level of nesting depth
% This provides a means by which the user can
% alter the effective \IAENGlabelindent for deeper
% levels
% There may not be such a thing as correct "standard IAENG"
% values. What IAENG actually does may depend on the specific
% circumstances.
% The first list level almost always has full indention.
% The second levels I've seen have only 75% of the normal indentation
% Three level or greater nestings are very rare. I am guessing
% that they don't use any indentation.
\def\IAENGlabelindentfactori{1.0}   % almost always one
\def\IAENGlabelindentfactorii{0.75} % 0.0 or 1.0 may be used in some cases
\def\IAENGlabelindentfactoriii{0.0} % 0.75? 0.5? 0.0?
\def\IAENGlabelindentfactoriv{0.0}
\def\IAENGlabelindentfactorv{0.0}
\def\IAENGlabelindentfactorvi{0.0}

% value actually used within IED lists, it is auto
% set to one of the 6 values above
% global changes here have no effect
\def\IAENGlabelindentfactor{1.0}

% This controls the default spacing between the end of the IED
% list labels and the list text, when normal text is used for
% the labels.
\newdimen\IAENGiednormlabelsep
\IAENGiednormlabelsep 0.6em

% This controls the default spacing between the end of the IED
% list labels and the list text, when math symbols are used for
% the labels (nomenclature lists). IAENG usually increases the 
% spacing in these cases
\newdimen\IAENGiedmathlabelsep
\IAENGiedmathlabelsep 1.2em

% This controls the extra vertical separation put above and
% below each IED list. IAENG usually puts a little extra spacing
% around each list. However, this spacing is barely noticeable.
\newskip\IAENGiedtopsep
\IAENGiedtopsep 2pt plus 1pt minus 1pt


% This command is executed within each IED list environment
% at the beginning of the list. You can use this to set the 
% parameters for some/all your IED list(s) without disturbing 
% global parameters that affect things other than lists.
% i.e., renewcommand{\IAENGiedlistdecl}{\setlength{\labelsep}{5em}}
% will alter the \labelsep for the next list(s) until 
% \IAENGiedlistdecl is redefined. 
\def\IAENGiedlistdecl{\relax}

% This command provides an easy way to set \leftmargin based
% on the \labelwidth, \labelsep and the argument \IAENGlabelindent
% Usage: \IAENGcalcleftmargin{width-to-indent-the-label}
% output is in the \leftmargin variable, i.e., effectively:
% \leftmargin = argument + \labelwidth + \labelsep
% Note controlled spacing here, shield end of lines with %
\def\IAENGcalcleftmargin#1{\setlength{\leftmargin}{#1}%
\addtolength{\leftmargin}{\labelwidth}%
\addtolength{\leftmargin}{\labelsep}}

% This command provides an easy way to set \labelwidth to the
% width of the given text. It is the same as
% \settowidth{\labelwidth}{label-text}
% and useful as a shorter alternative.
% Typically used to set \labelwidth to be the width
% of the longest label in the list
\def\IAENGsetlabelwidth#1{\settowidth{\labelwidth}{#1}}

% When this command is executed, IED lists will use the 
% IAENGiedmathlabelsep label separation rather than the normal
% spacing. To have an effect, this command must be executed via
% the \IAENGiedlistdecl or within the option of the IED list
% environments.
\def\IAENGusemathlabelsep{\setlength{\labelsep}{\IAENGiedmathlabelsep}}

% A flag which controls whether the IED lists automatically
% calculate \leftmargin from \IAENGlabelindent, \labelwidth and \labelsep
% Useful if you want to specify your own \leftmargin
% This flag must be set (\IAENGnocalcleftmargintrue or \IAENGnocalcleftmarginfalse) 
% via the \IAENGiedlistdecl or within the option of the IED list
% environments to have an effect.
\newif\ifIAENGnocalcleftmargin
\IAENGnocalcleftmarginfalse

% A flag which controls whether \IAENGlabelindent is multiplied by
% the \IAENGlabelindentfactor for each list level.
% This flag must be set via the \IAENGiedlistdecl or within the option 
% of the IED list environments to have an effect.
\newif\ifIAENGnolabelindentfactor
\IAENGnolabelindentfactorfalse


% internal variable to indicate type of IED label
% justification
% 0 - left; 1 - center; 2 - right
\def\@IAENGiedjustify{0}


% commands to allow the user to control IED
% label justifications. Use these commands within
% the IED environment option or in the \IAENGiedlistdecl
% Note that changing the normal list justifications
% is nonstandard and IAENG may not like it if you do so!
% I include these commands as they may be helpful to
% those who are using these enhanced list controls for
% other non-IAENG related LaTeX work.
% itemize and enumerate automatically default to right
% justification, description defaults to left.
\def\IAENGiedlabeljustifyl{\def\@IAENGiedjustify{0}}%left
\def\IAENGiedlabeljustifyc{\def\@IAENGiedjustify{1}}%center
\def\IAENGiedlabeljustifyr{\def\@IAENGiedjustify{2}}%right




% commands to save to and restore from the list parameter copies
% this allows us to set all the list parameters within
% the list_decl and prevent \list (and its \@list) 
% from overriding any of our parameters
% V1.6 use \edefs instead of dimen's to conserve dimen registers
% Note controlled spacing here, shield end of lines with %
\def\@IAENGsavelistparams{\edef\@IAENGiedtopsep{\the\topsep}%
\edef\@IAENGiedlabelwidth{\the\labelwidth}%
\edef\@IAENGiedlabelsep{\the\labelsep}%
\edef\@IAENGiedleftmargin{\the\leftmargin}%
\edef\@IAENGiedpartopsep{\the\partopsep}%
\edef\@IAENGiedparsep{\the\parsep}%
\edef\@IAENGieditemsep{\the\itemsep}%
\edef\@IAENGiedrightmargin{\the\rightmargin}%
\edef\@IAENGiedlistparindent{\the\listparindent}%
\edef\@IAENGieditemindent{\the\itemindent}}

% Note controlled spacing here
\def\@IAENGrestorelistparams{\topsep\@IAENGiedtopsep\relax%
\labelwidth\@IAENGiedlabelwidth\relax%
\labelsep\@IAENGiedlabelsep\relax%
\leftmargin\@IAENGiedleftmargin\relax%
\partopsep\@IAENGiedpartopsep\relax%
\parsep\@IAENGiedparsep\relax%
\itemsep\@IAENGieditemsep\relax%
\rightmargin\@IAENGiedrightmargin\relax%
\listparindent\@IAENGiedlistparindent\relax%
\itemindent\@IAENGieditemindent\relax}


% v1.6b provide original LaTeX IED list environments
% note that latex.ltx defines \itemize and \enumerate, but not \description
% which must be created by the base classes
% save original LaTeX itemize and enumerate
\let\LaTeXitemize\itemize
\let\endLaTeXitemize\enditemize
\let\LaTeXenumerate\enumerate
\let\endLaTeXenumerate\endenumerate

% provide original LaTeX description environment from article.cls
\newenvironment{LaTeXdescription}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                 \normalfont\bfseries #1}


% override LaTeX's default IED lists
\def\itemize{\@IAENGitemize}
\def\enditemize{\@endIAENGitemize}
\def\enumerate{\@IAENGenumerate}
\def\endenumerate{\@endIAENGenumerate}
\def\description{\@IAENGdescription}
\def\enddescription{\@endIAENGdescription}

% provide the user with aliases - may help those using packages that
% override itemize, enumerate, or description
\def\IAENGitemize{\@IAENGitemize}
\def\endIAENGitemize{\@endIAENGitemize}
\def\IAENGenumerate{\@IAENGenumerate}
\def\endIAENGenumerate{\@endIAENGenumerate}
\def\IAENGdescription{\@IAENGdescription}
\def\endIAENGdescription{\@endIAENGdescription}


% V1.6 we want to keep the IAENGtran IED list definitions as our own internal
% commands so they are protected against redefinition
\def\@IAENGitemize{\@ifnextchar[{\@@IAENGitemize}{\@@IAENGitemize[\relax]}}
\def\@IAENGenumerate{\@ifnextchar[{\@@IAENGenumerate}{\@@IAENGenumerate[\relax]}}
\def\@IAENGdescription{\@ifnextchar[{\@@IAENGdescription}{\@@IAENGdescription[\relax]}}
\def\@endIAENGitemize{\endlist}
\def\@endIAENGenumerate{\endlist}
\def\@endIAENGdescription{\endlist}


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IAENGtran itemized list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IAENGitemize[#1]{%
                \ifnum\@itemdepth>3\relax\@toodeep\else%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                \advance\@itemdepth\@ne%
                \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
                % get the labelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IAENGlabelindentfactor{\csname IAENGlabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IAENGiedjustify{2}% right justified labels are default
                % set other defaults
                \IAENGnocalcleftmarginfalse%
                \IAENGnolabelindentfactorfalse%
                \topsep\IAENGiedtopsep%
                \IAENGlabelindent\IAENGilabelindent%
                \labelsep\IAENGiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % calculate the label width
                % the user can override this later if
                % they specified a \labelwidth
                \settowidth{\labelwidth}{\csname labelitem\romannumeral\the\@itemdepth\endcsname}%
                \@IAENGsavelistparams% save our list parameters
                \list{\csname\@itemitem\endcsname}{%
                \@IAENGrestorelistparams% override any list{} changes
                                       % to our globals
                \let\makelabel\@IAENGiedmakelabel% v1.6b setup \makelabel
                \IAENGiedlistdecl% let user alter parameters
                #1\relax%
                % If the user has requested not to use the
                % labelindent factor, don't revise \labelindent
                \ifIAENGnolabelindentfactor\relax%
                \else\IAENGlabelindent=\IAENGlabelindentfactor\labelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IAENGlabelindent, \labelwidth and
                % \labelsep
                \ifIAENGnocalcleftmargin\relax%
                \else\IAENGcalcleftmargin{\IAENGlabelindent}%
                \fi}\fi\fi}%


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IAENGtran enumerate list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IAENGenumerate[#1]{%
                \ifnum\@enumdepth>3\relax\@toodeep\else%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                \advance\@enumdepth\@ne%
                \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
                % get the labelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IAENGlabelindentfactor{\csname IAENGlabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IAENGiedjustify{2}% right justified labels are default
                % set other defaults
                \IAENGnocalcleftmarginfalse%
                \IAENGnolabelindentfactorfalse%
                \topsep\IAENGiedtopsep%
                \IAENGlabelindent\IAENGelabelindent%
                \labelsep\IAENGiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % calculate the label width
                % We'll set it to the width suitable for all labels using
                % normalfont 1) to 9)
                % The user can override this later
                \settowidth{\labelwidth}{9)}%
                \@IAENGsavelistparams% save our list parameters
                \list{\csname label\@enumctr\endcsname}{\usecounter{\@enumctr}%
                \@IAENGrestorelistparams% override any list{} changes
                                       % to our globals
                \let\makelabel\@IAENGiedmakelabel% v1.6b setup \makelabel
                \IAENGiedlistdecl% let user alter parameters 
                #1\relax%
                % If the user has requested not to use the
                % IAENGlabelindent factor, don't revise \IAENGlabelindent
                \ifIAENGnolabelindentfactor\relax%
                \else\IAENGlabelindent=\IAENGlabelindentfactor\IAENGlabelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IAENGlabelindent, \labelwidth and
                % \labelsep
                \ifIAENGnocalcleftmargin\relax%
                \else\IAENGcalcleftmargin{\IAENGlabelindent}%
                \fi}\fi\fi}%


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IAENGtran description list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IAENGdescription[#1]{%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                % get the labelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IAENGlabelindentfactor{\csname IAENGlabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IAENGiedjustify{0}% left justified labels are default
                % set other defaults
                \IAENGnocalcleftmarginfalse%
                \IAENGnolabelindentfactorfalse%
                \topsep\IAENGiedtopsep% 
                \IAENGlabelindent\IAENGdlabelindent%
                % assume normal labelsep
                \labelsep\IAENGiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % Bogus label width in case the user forgets
                % to set it.
                % TIP: If you want to see what a variable's width is you
                % can use the TeX command \showthe\width-variable to 
                % display it on the screen during compilation 
                % (This might be helpful to know when you need to find out
                % which label is the widest)
                \settowidth{\labelwidth}{Hello}%
                \@IAENGsavelistparams% save our list parameters
                \list{}{\@IAENGrestorelistparams% override any list{} changes
                                               % to our globals
                \let\makelabel\@IAENGiedmakelabel% v1.6b setup \makelabel
                \IAENGiedlistdecl% let user alter parameters 
                #1\relax%
                % If the user has requested not to use the
                % labelindent factor, don't revise \IAENGlabelindent
                \ifIAENGnolabelindentfactor\relax%
                \else\IAENGlabelindent=\IAENGlabelindentfactor\IAENGlabelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IAENGlabelindent, \labelwidth and
                % \labelsep
                \ifIAENGnocalcleftmargin\relax%
                \else\IAENGcalcleftmargin{\IAENGlabelindent}\relax%
                \fi}\fi}

% v1.6b we use one makelabel that does justification as needed.
\def\@IAENGiedmakelabel#1{\relax\if\@IAENGiedjustify 0\relax
\makebox[\labelwidth][l]{\normalfont #1}\else
\if\@IAENGiedjustify 1\relax
\makebox[\labelwidth][c]{\normalfont #1}\else
\makebox[\labelwidth][r]{\normalfont #1}\fi\fi}


% VERSE and QUOTE
% V1.7 define environments with newenvironment
\newenvironment{verse}{\let\\=\@centercr
    \list{}{\itemsep\z@ \itemindent -1.5em \listparindent \itemindent
    \rightmargin\leftmargin\advance\leftmargin 1.5em}\item\relax}
    {\endlist}
\newenvironment{quotation}{\list{}{\listparindent 1.5em \itemindent\listparindent
    \rightmargin\leftmargin \parsep 0pt plus 1pt}\item\relax}
    {\endlist}
\newenvironment{quote}{\list{}{\rightmargin\leftmargin}\item\relax}
    {\endlist}


% \titlepage
% provided only for backward compatibility. \maketitle is the correct
% way to create the title page. 
\newif\if@restonecol
\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
    \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

% standard values from article.cls
\arraycolsep     5pt
\arrayrulewidth .4pt
\doublerulesep   2pt

\tabcolsep       6pt
\tabbingsep      0.5em


%% FOOTNOTES
%
%\skip\footins 10pt plus 4pt minus 2pt
% V1.6 respond to changes in font size
% space added above the footnotes (if present)
\skip\footins 0.9\baselineskip  plus 0.4\baselineskip  minus 0.2\baselineskip

% V1.6, we need to make \footnotesep responsive to changes
% in \baselineskip or strange spacings will result when in
% draft mode. Here is a little LaTeX secret - \footnotesep
% determines the height of an invisible strut that is placed
% *above* the baseline of footnotes after the first. Since
% LaTeX considers the space for characters to be 0.7/baselineskip
% above the baseline and 0.3/baselineskip below it, we need to
% use 0.7/baselineskip as a \footnotesep to maintain equal spacing
% between all the lines of the footnotes. IAENG often uses a tad
% more, so use 0.8\baselineskip. This slightly larger value also helps
% the text to clear the footnote marks. Note that \thanks in IAENGtran
% uses its own value of \footnotesep which is set in \maketitle.
{\footnotesize
\global\footnotesep 0.8\baselineskip}


\skip\@mpfootins = \skip\footins
\fboxsep = 3pt
\fboxrule = .4pt
% V1.6 use 1em, then use LaTeX2e's \@makefnmark
% Note that IAENG normally *left* aligns the footnote marks, so we don't need
% box resizing tricks here.
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@makefnmark}#1}% V1.6 use 1em
% V1.7 compsoc does not use superscipts for footnote marks
\ifCLASSOPTIONcompsoc
\def\@IAENGcompsocmakefnmark{\hbox{\normalfont\@thefnmark.\ }}
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@IAENGcompsocmakefnmark}#1}
\fi

% IAENG does not use footnote rules
\def\footnoterule{}

% V1.7 for compsoc, IAENG uses a footnote rule only for \thanks. We devise a "one-shot"
% system to implement this.
\newif\if@IAENGenableoneshotfootnoterule
\@IAENGenableoneshotfootnoterulefalse
\ifCLASSOPTIONcompsoc
\def\footnoterule{\relax\if@IAENGenableoneshotfootnoterule
\kern-5pt
\hbox to \columnwidth{\hfill\vrule width 0.5\columnwidth height 0.4pt\hfill}
\kern4.6pt
\global\@IAENGenableoneshotfootnoterulefalse
\else
\relax
\fi}
\fi

% V1.6 do not allow LaTeX to break a footnote across multiple pages
\interfootnotelinepenalty=10000

% V1.6 discourage breaks within equations
% Note that amsmath normally sets this to 10000,
% but LaTeX2e normally uses 100.
\interdisplaylinepenalty=2500

% default allows section depth up to /paragraph
\setcounter{secnumdepth}{4}

% technotes do not allow /paragraph
\ifCLASSOPTIONtechnote
   \setcounter{secnumdepth}{3}
\fi
% neither do compsoc conferences
\@IAENGcompsocconfonly{\setcounter{secnumdepth}{3}}


\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]

% used only by IAENGtran's IAENGeqnarray as other packages may
% have their own, different, implementations
\newcounter{IAENGsubequation}[equation]

% as shown when called by user from \ref, \label and in table of contents
\def\theequation{\arabic{equation}}                          % 1
\def\theIAENGsubequation{\theequation\alph{IAENGsubequation}}  % 1a (used only by IAENGtran's IAENGeqnarray)
\ifCLASSOPTIONcompsoc
% compsoc is all arabic
\def\thesection{\arabic{section}}                
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\else
\def\thesection{\Roman{section}}                             % I
% V1.7, \mbox prevents breaks around - 
\def\thesubsection{\mbox{\thesection-\Alph{subsection}}}     % I-A
% V1.7 use I-A1 format used by IAENG rather than I-A.1
\def\thesubsubsection{\thesubsection\arabic{subsubsection}}  % I-A1
\def\theparagraph{\thesubsubsection\alph{paragraph}}         % I-A1a
\fi

% From Heiko Oberdiek. Because of the \mbox in \thesubsection, we need to
% tell hyperref to disable the \mbox command when making PDF bookmarks.
% This done already with hyperref.sty version 6.74o and later, but
% it will not hurt to do it here again for users of older versions.
\@ifundefined{pdfstringdefPreHook}{\let\pdfstringdefPreHook\@empty}{}%
\g@addto@macro\pdfstringdefPreHook{\let\mbox\relax}


% Main text forms (how shown in main text headings)
% V1.6, using \thesection in \thesectiondis allows changes
% in the former to automatically appear in the latter
\ifCLASSOPTIONcompsoc
  \ifCLASSOPTIONconference% compsoc conference
    \def\thesectiondis{\thesection.}
    \def\thesubsectiondis{\thesectiondis\arabic{subsection}.}
    \def\thesubsubsectiondis{\thesubsectiondis\arabic{subsubsection}.}
    \def\theparagraphdis{\thesubsubsectiondis\arabic{paragraph}.}
  \else% compsoc not conferencs
    \def\thesectiondis{\thesection}
    \def\thesubsectiondis{\thesectiondis.\arabic{subsection}}
    \def\thesubsubsectiondis{\thesubsectiondis.\arabic{subsubsection}}
    \def\theparagraphdis{\thesubsubsectiondis.\arabic{paragraph}}
  \fi
\else% not compsoc
  \def\thesectiondis{\thesection.}                   % I.
  \def\thesubsectiondis{\Alph{subsection}.}          % B.
  \def\thesubsubsectiondis{\arabic{subsubsection})}  % 3)
  \def\theparagraphdis{\alph{paragraph})}            % d)
\fi

% just like LaTeX2e's \@eqnnum
\def\theequationdis{{\normalfont \normalcolor (\theequation)}}% (1)
% IAENGsubequation used only by IAENGtran's IAENGeqnarray
\def\theIAENGsubequationdis{{\normalfont \normalcolor (\theIAENGsubequation)}}% (1a)
% redirect LaTeX2e's equation number display and all that depend on
% it, through IAENGtran's \theequationdis
\def\@eqnnum{\theequationdis}



% V1.7 provide string macros as article.cls does
\def\contentsname{Contents}
\def\listfigurename{List of Figures}
\def\listtablename{List of Tables}
\def\refname{References}
\def\indexname{Index}
\def\figurename{Fig.}
\def\tablename{TABLE}
\@IAENGcompsocconfonly{\def\figurename{Figure}\def\tablename{Table}}
\def\partname{Part}
\def\appendixname{Appendix}
\def\abstractname{Abstract}
% IAENG specific names
\def\IAENGkeywordsname{Index Terms}
\def\IAENGproofname{Proof}


% LIST OF FIGURES AND TABLES AND TABLE OF CONTENTS
%
\def\@pnumwidth{1.55em}
\def\@tocrmarg{2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}

% adjusted some spacings here so that section numbers will not easily 
% collide with the section titles. 
% VIII; VIII-A; and VIII-A.1 are usually the worst offenders.
% MDS 1/2001
\def\tableofcontents{\section*{\contentsname}\@starttoc{toc}}
\def\l@section#1#2{\addpenalty{\@secpenalty}\addvspace{1.0em plus 1pt}%
    \@tempdima 2.75em \begingroup \parindent \z@ \rightskip \@pnumwidth%
    \parfillskip-\@pnumwidth {\bfseries\leavevmode #1}\hfil\hbox to\@pnumwidth{\hss #2}\par%
    \endgroup}
% argument format #1:level, #2:labelindent,#3:labelsep
\def\l@subsection{\@dottedtocline{2}{2.75em}{3.75em}}
\def\l@subsubsection{\@dottedtocline{3}{6.5em}{4.5em}}
% must provide \l@ defs for ALL sublevels EVEN if tocdepth
% is such as they will not appear in the table of contents
% these defs are how TOC knows what level these things are!
\def\l@paragraph{\@dottedtocline{4}{6.5em}{5.5em}}
\def\l@subparagraph{\@dottedtocline{5}{6.5em}{6.5em}}
\def\listoffigures{\section*{\listfigurename}\@starttoc{lof}}
\def\l@figure{\@dottedtocline{1}{0em}{2.75em}}
\def\listoftables{\section*{\listtablename}\@starttoc{lot}}
\let\l@table\l@figure


%% Definitions for floats
%%
%% Normal Floats
\floatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\textfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
\@fptop 0pt plus 1fil
\@fpsep 0.75\baselineskip plus 2fil 
\@fpbot 0pt plus 1fil
\def\topfraction{0.9}
\def\bottomfraction{0.4}
\def\floatpagefraction{0.8}
% V1.7, let top floats approach 90% of page
\def\textfraction{0.1}

%% Double Column Floats
\dblfloatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip

\dbltextfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
% Note that it would be nice if the rubber here actually worked in LaTeX2e.
% There is a long standing limitation in LaTeX, first discovered (to the best
% of my knowledge) by Alan Jeffrey in 1992. LaTeX ignores the stretchable
% portion of \dbltextfloatsep, and as a result, double column figures can and
% do result in an non-integer number of lines in the main text columns with
% underfull vbox errors as a consequence. A post to comp.text.tex
% by Donald Arseneau confirms that this had not yet been fixed in 1998.
% IAENGtran V1.6 will fix this problem for you in the titles, but it doesn't
% protect you from other double floats. Happy vspace'ing.

\@dblfptop 0pt plus 1fil
\@dblfpsep 0.75\baselineskip plus 2fil
\@dblfpbot 0pt plus 1fil
\def\dbltopfraction{0.8}
\def\dblfloatpagefraction{0.8}
\setcounter{dbltopnumber}{4}

\intextsep 1\baselineskip plus 0.2\baselineskip minus  0.2\baselineskip
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}



% article class provides these, we should too.
\newlength\abovecaptionskip
\newlength\belowcaptionskip
% but only \abovecaptionskip is used above figure captions and *below* table
% captions
\setlength\abovecaptionskip{0.5\baselineskip}
\setlength\belowcaptionskip{0pt}
% V1.6 create hooks in case the caption spacing ever needs to be
% overridden by a user
\def\@IAENGfigurecaptionsepspace{\vskip\abovecaptionskip\relax}%
\def\@IAENGtablecaptionsepspace{\vskip\abovecaptionskip\relax}%


% 1.6b revise caption system so that \@makecaption uses two arguments
% as with LaTeX2e. Otherwise, there will be problems when using hyperref.
\def\@IAENGtablestring{table}

\ifCLASSOPTIONcompsoc
% V1.7 compsoc \@makecaption
\ifCLASSOPTIONconference% compsoc conference
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IAENGtablestring%
% if a table, do table caption
\normalsize\begin{center}{\normalfont\sffamily\normalsize {#1.}~ #2}\end{center}%
\@IAENGtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IAENGfigurecaptionsepspace
\setbox\@tempboxa\hbox{\normalfont\sffamily\normalsize {#1.}~ #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\sffamily\normalsize {#1.}~ }%
\parbox[t]{\hsize}{\normalfont\sffamily\normalsize \noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, center
\else%
\hbox to\hsize{\normalfont\sffamily\normalsize\hfil\box\@tempboxa\hfil}%
\fi\fi}
\else% nonconference compsoc
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IAENGtablestring%
% if a table, do table caption
\normalsize\begin{center}{\normalfont\sffamily\normalsize #1}\\{\normalfont\sffamily\normalsize #2}\end{center}%
\@IAENGtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IAENGfigurecaptionsepspace
\setbox\@tempboxa\hbox{\normalfont\sffamily\normalsize {#1.}~ #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\sffamily\normalsize {#1.}~ }%
\parbox[t]{\hsize}{\normalfont\sffamily\normalsize \noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, left justify
\else%
\hbox to\hsize{\normalfont\sffamily\normalsize\box\@tempboxa\hfil}%
\fi\fi}
\fi

\else% traditional noncompsoc \@makecaption
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IAENGtablestring%
% if a table, do table caption
\footnotesize\begin{center}{\normalfont\footnotesize #1}\\{\normalfont\footnotesize\scshape #2}\end{center}%
\@IAENGtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IAENGfigurecaptionsepspace
% 3/2001 use footnotesize, not small; use two nonbreaking spaces, not one
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}~~ #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}~~ }%
\parbox[t]{\hsize}{\normalfont\footnotesize\noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, center if conference, left justify otherwise
\else%
\ifCLASSOPTIONconference \hbox to\hsize{\normalfont\footnotesize\hfil\box\@tempboxa\hfil}%
\else \hbox to\hsize{\normalfont\footnotesize\box\@tempboxa\hfil}%
\fi\fi\fi}
\fi



% V1.7 disable captions class option, do so in a way that retains operation of \label
% within \caption
\ifCLASSOPTIONcaptionsoff
\long\def\@makecaption#1#2{\vspace*{2em}\footnotesize\begin{center}{\footnotesize #1}\end{center}%
\let\@IAENGtemporiglabeldefsave\label
\let\@IAENGtemplabelargsave\relax
\def\label##1{\gdef\@IAENGtemplabelargsave{##1}}%
\setbox\@tempboxa\hbox{#2}%
\let\label\@IAENGtemporiglabeldefsave
\ifx\@IAENGtemplabelargsave\relax\else\label{\@IAENGtemplabelargsave}\fi}
\fi


% V1.7 define end environments with \def not \let so as to work OK with
% preview-latex
\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\def\figure{\@float{figure}}
\def\endfigure{\end@float}
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}
\newcounter{table}
\ifCLASSOPTIONcompsoc
\def\thetable{\arabic{table}}
\else
\def\thetable{\@Roman\c@table}
\fi
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
% V1.6 IAENG uses 8pt text for tables
% to default to footnotesize, we hack into LaTeX2e's \@floatboxreset and pray
\def\table{\def\@floatboxreset{\reset@font\footnotesize\@setminipage}\@float{table}}
\def\endtable{\end@float}
% v1.6b double column tables need to default to footnotesize as well.
\@namedef{table*}{\def\@floatboxreset{\reset@font\footnotesize\@setminipage}\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}




%%
%% START OF IAENGeqnarry DEFINITIONS
%%
%% Inspired by the concepts, examples, and previous works of LaTeX 
%% coders and developers such as Donald Arseneau, Fred Bartlett, 
%% David Carlisle, Tony Liu, Frank Mittelbach, Piet van Oostrum, 
%% Roland Winkler and Mark Wooding.
%% I don't make the claim that my work here is even near their calibre. ;)


% hook to allow easy changeover to IAENGtran.cls/tools.sty error reporting
\def\@IAENGclspkgerror{\ClassError{IAENGtran}}

\newif\if@IAENGeqnarraystarform% flag to indicate if the environment was called as the star form
\@IAENGeqnarraystarformfalse

\newif\if@advanceIAENGeqncolcnt% tracks if the environment should advance the col counter
% allows a way to make an \IAENGeqnarraybox that can be used within an \IAENGeqnarray
% used by IAENGeqnarraymulticol so that it can work properly in both
\@advanceIAENGeqncolcnttrue

\newcount\@IAENGeqnnumcols % tracks how many IAENGeqnarray cols are defined
\newcount\@IAENGeqncolcnt  % tracks how many IAENGeqnarray cols the user actually used


% The default math style used by the columns
\def\IAENGeqnarraymathstyle{\displaystyle}
% The default text style used by the columns
% default to using the current font
\def\IAENGeqnarraytextstyle{\relax}

% like the iedlistdecl but for \IAENGeqnarray
\def\IAENGeqnarraydecl{\relax}
\def\IAENGeqnarrayboxdecl{\relax}

% \yesnumber is the opposite of \nonumber
% a novel concept with the same def as the equationarray package
% However, we give IAENG versions too since some LaTeX packages such as 
% the MDWtools mathenv.sty redefine \nonumber to something else.
\providecommand{\yesnumber}{\global\@eqnswtrue}
\def\IAENGyesnumber{\global\@eqnswtrue}
\def\IAENGnonumber{\global\@eqnswfalse}


\def\IAENGyessubnumber{\global\@IAENGissubequationtrue\global\@eqnswtrue%
\if@IAENGeqnarrayISinner% only do something inside an IAENGeqnarray
\if@IAENGlastlinewassubequation\addtocounter{equation}{-1}\else\setcounter{IAENGsubequation}{1}\fi%
\def\@currentlabel{\p@IAENGsubequation\theIAENGsubequation}\fi}

% flag to indicate that an equation is a sub equation
\newif\if@IAENGissubequation%
\@IAENGissubequationfalse

% allows users to "push away" equations that get too close to the equation numbers
\def\IAENGeqnarraynumspace{\hphantom{\if@IAENGissubequation\theIAENGsubequationdis\else\theequationdis\fi}}

% provides a way to span multiple columns within IAENGeqnarray environments
% will consider \if@advanceIAENGeqncolcnt before globally advancing the
% column counter - so as to work within \IAENGeqnarraybox
% usage: \IAENGeqnarraymulticol{number cols. to span}{col type}{cell text}
\long\def\IAENGeqnarraymulticol#1#2#3{\multispan{#1}%
% check if column is defined
\relax\expandafter\ifx\csname @IAENGeqnarraycolDEF#2\endcsname\@IAENGeqnarraycolisdefined%
\csname @IAENGeqnarraycolPRE#2\endcsname#3\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IAENGeqnarraycolPOST#2\endcsname%
\else% if not, error and use default type
\@IAENGclspkgerror{Invalid column type "#2" in \string\IAENGeqnarraymulticol.\MessageBreak
Using a default centering column instead}%
{You must define IAENGeqnarray column types before use.}%
\csname @IAENGeqnarraycolPRE@IAENGdefault\endcsname#3\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IAENGeqnarraycolPOST@IAENGdefault\endcsname%
\fi%
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by #1\relax\fi}

% like \omit, but maintains track of the column counter for \IAENGeqnarray
\def\IAENGeqnarrayomit{\omit\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by 1\relax\fi}


% provides a way to define a letter referenced column type
% usage: \IAENGeqnarraydefcol{col. type letter/name}{pre insertion text}{post insertion text}
\def\IAENGeqnarraydefcol#1#2#3{\expandafter\def\csname @IAENGeqnarraycolPRE#1\endcsname{#2}%
\expandafter\def\csname @IAENGeqnarraycolPOST#1\endcsname{#3}%
\expandafter\def\csname @IAENGeqnarraycolDEF#1\endcsname{1}}


% provides a way to define a numerically referenced inter-column glue types
% usage: \IAENGeqnarraydefcolsep{col. glue number}{glue definition}
\def\IAENGeqnarraydefcolsep#1#2{\expandafter\def\csname @IAENGeqnarraycolSEP\romannumeral #1\endcsname{#2}%
\expandafter\def\csname @IAENGeqnarraycolSEPDEF\romannumeral #1\endcsname{1}}


\def\@IAENGeqnarraycolisdefined{1}% just a macro for 1, used for checking undefined column types


% expands and appends the given argument to the \@IAENGtrantmptoksA token list
% used to build up the \halign preamble
\def\@IAENGappendtoksA#1{\edef\@@IAENGappendtoksA{\@IAENGtrantmptoksA={\the\@IAENGtrantmptoksA #1}}%
\@@IAENGappendtoksA}

% also appends to \@IAENGtrantmptoksA, but does not expand the argument
% uses \toks8 as a scratchpad register
\def\@IAENGappendNOEXPANDtoksA#1{\toks8={#1}%
\edef\@@IAENGappendNOEXPANDtoksA{\@IAENGtrantmptoksA={\the\@IAENGtrantmptoksA\the\toks8}}%
\@@IAENGappendNOEXPANDtoksA}

% define some common column types for the user
% math
\IAENGeqnarraydefcol{l}{$\IAENGeqnarraymathstyle}{$\hfil}
\IAENGeqnarraydefcol{c}{\hfil$\IAENGeqnarraymathstyle}{$\hfil}
\IAENGeqnarraydefcol{r}{\hfil$\IAENGeqnarraymathstyle}{$}
\IAENGeqnarraydefcol{L}{$\IAENGeqnarraymathstyle{}}{{}$\hfil}
\IAENGeqnarraydefcol{C}{\hfil$\IAENGeqnarraymathstyle{}}{{}$\hfil}
\IAENGeqnarraydefcol{R}{\hfil$\IAENGeqnarraymathstyle{}}{{}$}
% text
\IAENGeqnarraydefcol{s}{\IAENGeqnarraytextstyle}{\hfil}
\IAENGeqnarraydefcol{t}{\hfil\IAENGeqnarraytextstyle}{\hfil}
\IAENGeqnarraydefcol{u}{\hfil\IAENGeqnarraytextstyle}{}

% vertical rules
\IAENGeqnarraydefcol{v}{}{\vrule width\arrayrulewidth}
\IAENGeqnarraydefcol{vv}{\vrule width\arrayrulewidth\hfil}{\hfil\vrule width\arrayrulewidth}
\IAENGeqnarraydefcol{V}{}{\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth}
\IAENGeqnarraydefcol{VV}{\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth\hfil}%
{\hfil\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth}

% horizontal rules
\IAENGeqnarraydefcol{h}{}{\leaders\hrule height\arrayrulewidth\hfil}
\IAENGeqnarraydefcol{H}{}{\leaders\vbox{\hrule width\arrayrulewidth\vskip\doublerulesep\hrule width\arrayrulewidth}\hfil}

% plain
\IAENGeqnarraydefcol{x}{}{}
\IAENGeqnarraydefcol{X}{$}{$}

% the default column type to use in the event a column type is not defined
\IAENGeqnarraydefcol{@IAENGdefault}{\hfil$\IAENGeqnarraymathstyle}{$\hfil}


% a zero tabskip (used for "-" col types)
\def\@IAENGeqnarraycolSEPzero{0pt plus 0pt minus 0pt}
% a centering tabskip (used for "+" col types)
\def\@IAENGeqnarraycolSEPcenter{1000pt plus 0pt minus 1000pt}

% top level default tabskip glues for the start, end, and inter-column
% may be reset within environments not always at the top level, e.g., \IAENGeqnarraybox
\edef\@IAENGeqnarraycolSEPdefaultstart{\@IAENGeqnarraycolSEPcenter}% default start glue
\edef\@IAENGeqnarraycolSEPdefaultend{\@IAENGeqnarraycolSEPcenter}% default end glue
\edef\@IAENGeqnarraycolSEPdefaultmid{\@IAENGeqnarraycolSEPzero}% default inter-column glue



% creates a vertical rule that extends from the bottom to the top a a cell
% Provided in case other packages redefine \vline some other way.
% usage: \IAENGeqnarrayvrule[rule thickness]
% If no argument is provided, \arrayrulewidth will be used for the rule thickness. 
\newcommand\IAENGeqnarrayvrule[1][\arrayrulewidth]{\vrule\@width#1\relax}

% creates a blank separator row
% usage: \IAENGeqnarrayseprow[separation length][font size commands]
% default is \IAENGeqnarrayseprow[0.25\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \skip5 as a scratch register - calls \@IAENGeqnarraystrutsize which uses more scratch registers
\def\IAENGeqnarrayseprow{\relax\@ifnextchar[{\@IAENGeqnarrayseprow}{\@IAENGeqnarrayseprow[0.25\normalbaselineskip]}}
\def\@IAENGeqnarrayseprow[#1]{\relax\@ifnextchar[{\@@IAENGeqnarrayseprow[#1]}{\@@IAENGeqnarrayseprow[#1][\relax]}}
\def\@@IAENGeqnarrayseprow[#1][#2]{\def\@IAENGeqnarrayseprowARGONE{#1}%
\ifx\@IAENGeqnarrayseprowARGONE\@empty%
% get the skip value, based on the font commands
% use skip5 because \IAENGeqnarraystrutsize uses \skip0, \skip2, \skip3
% assign within a bogus box to confine the font changes
{\setbox0=\hbox{#2\relax\global\skip5=0.25\normalbaselineskip}}%
\else%
{\setbox0=\hbox{#2\relax\global\skip5=#1}}%
\fi%
\@IAENGeqnarrayhoptolastcolumn\IAENGeqnarraystrutsize{\skip5}{0pt}[\relax]\relax}

% creates a blank separator row, but omits all the column templates
% usage: \IAENGeqnarrayseprowcut[separation length][font size commands]
% default is \IAENGeqnarrayseprowcut[0.25\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \skip5 as a scratch register - calls \@IAENGeqnarraystrutsize which uses more scratch registers
\def\IAENGeqnarrayseprowcut{\multispan{\@IAENGeqnnumcols}\relax% span all the cols
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\@ifnextchar[{\@IAENGeqnarrayseprowcut}{\@IAENGeqnarrayseprowcut[0.25\normalbaselineskip]}}
\def\@IAENGeqnarrayseprowcut[#1]{\relax\@ifnextchar[{\@@IAENGeqnarrayseprowcut[#1]}{\@@IAENGeqnarrayseprowcut[#1][\relax]}}
\def\@@IAENGeqnarrayseprowcut[#1][#2]{\def\@IAENGeqnarrayseprowARGONE{#1}%
\ifx\@IAENGeqnarrayseprowARGONE\@empty%
% get the skip value, based on the font commands
% use skip5 because \IAENGeqnarraystrutsize uses \skip0, \skip2, \skip3
% assign within a bogus box to confine the font changes
{\setbox0=\hbox{#2\relax\global\skip5=0.25\normalbaselineskip}}%
\else%
{\setbox0=\hbox{#2\relax\global\skip5=#1}}%
\fi%
\IAENGeqnarraystrutsize{\skip5}{0pt}[\relax]\relax}



% draws a single rule across all the columns optional
% argument determines the rule width, \arrayrulewidth is the default
% updates column counter as needed and turns off struts
% usage: \IAENGeqnarrayrulerow[rule line thickness]
\def\IAENGeqnarrayrulerow{\multispan{\@IAENGeqnnumcols}\relax% span all the cols
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\@ifnextchar[{\@IAENGeqnarrayrulerow}{\@IAENGeqnarrayrulerow[\arrayrulewidth]}}
\def\@IAENGeqnarrayrulerow[#1]{\leaders\hrule height#1\hfil\relax% put in our rule 
% turn off any struts
\IAENGeqnarraystrutsize{0pt}{0pt}[\relax]\relax}


% draws a double rule by using a single rule row, a separator row, and then
% another single rule row 
% first optional argument determines the rule thicknesses, \arrayrulewidth is the default
% second optional argument determines the rule spacing, \doublerulesep is the default
% usage: \IAENGeqnarraydblrulerow[rule line thickness][rule spacing]
\def\IAENGeqnarraydblrulerow{\multispan{\@IAENGeqnnumcols}\relax% span all the cols
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\@ifnextchar[{\@IAENGeqnarraydblrulerow}{\@IAENGeqnarraydblrulerow[\arrayrulewidth]}}
\def\@IAENGeqnarraydblrulerow[#1]{\relax\@ifnextchar[{\@@IAENGeqnarraydblrulerow[#1]}%
{\@@IAENGeqnarraydblrulerow[#1][\doublerulesep]}}
\def\@@IAENGeqnarraydblrulerow[#1][#2]{\def\@IAENGeqnarraydblrulerowARG{#1}%
% we allow the user to say \IAENGeqnarraydblrulerow[][]
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\@IAENGeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IAENGeqnarrayrulerow[#1]\relax%
\fi%
\def\@IAENGeqnarraydblrulerowARG{#2}%
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\\\IAENGeqnarrayseprow[\doublerulesep][\relax]%
\else%
\\\IAENGeqnarrayseprow[#2][\relax]%
\fi%
\\\multispan{\@IAENGeqnnumcols}%
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\def\@IAENGeqnarraydblrulerowARG{#1}%
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\@IAENGeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IAENGeqnarrayrulerow[#1]%
\fi%
}

% draws a double rule by using a single rule row, a separator (cutting) row, and then
% another single rule row 
% first optional argument determines the rule thicknesses, \arrayrulewidth is the default
% second optional argument determines the rule spacing, \doublerulesep is the default
% usage: \IAENGeqnarraydblrulerow[rule line thickness][rule spacing]
\def\IAENGeqnarraydblrulerowcut{\multispan{\@IAENGeqnnumcols}\relax% span all the cols
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\@ifnextchar[{\@IAENGeqnarraydblrulerowcut}{\@IAENGeqnarraydblrulerowcut[\arrayrulewidth]}}
\def\@IAENGeqnarraydblrulerowcut[#1]{\relax\@ifnextchar[{\@@IAENGeqnarraydblrulerowcut[#1]}%
{\@@IAENGeqnarraydblrulerowcut[#1][\doublerulesep]}}
\def\@@IAENGeqnarraydblrulerowcut[#1][#2]{\def\@IAENGeqnarraydblrulerowARG{#1}%
% we allow the user to say \IAENGeqnarraydblrulerow[][]
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\@IAENGeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IAENGeqnarrayrulerow[#1]%
\fi%
\def\@IAENGeqnarraydblrulerowARG{#2}%
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\\\IAENGeqnarrayseprowcut[\doublerulesep][\relax]%
\else%
\\\IAENGeqnarrayseprowcut[#2][\relax]%
\fi%
\\\multispan{\@IAENGeqnnumcols}%
% advance column counter only if the IAENGeqnarray environment wants it
\if@advanceIAENGeqncolcnt\global\advance\@IAENGeqncolcnt by \@IAENGeqnnumcols\relax\fi%
\def\@IAENGeqnarraydblrulerowARG{#1}%
\ifx\@IAENGeqnarraydblrulerowARG\@empty%
\@IAENGeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IAENGeqnarrayrulerow[#1]%
\fi%
}



% inserts a full row's worth of &'s
% relies on \@IAENGeqnnumcols to provide the correct number of columns
% uses \@IAENGtrantmptoksA, \count0 as scratch registers
\def\@IAENGeqnarrayhoptolastcolumn{\@IAENGtrantmptoksA={}\count0=1\relax%
\loop% add cols if the user did not use them all
\ifnum\count0<\@IAENGeqnnumcols\relax%
\@IAENGappendtoksA{&}%
\advance\count0 by 1\relax% update the col count
\repeat%
\the\@IAENGtrantmptoksA%execute the &'s
}



\newif\if@IAENGeqnarrayISinner % flag to indicate if we are within the lines
\@IAENGeqnarrayISinnerfalse    % of an IAENGeqnarray - after the IAENGeqnarraydecl

\edef\@IAENGeqnarrayTHEstrutheight{0pt} % height and depth of IAENGeqnarray struts
\edef\@IAENGeqnarrayTHEstrutdepth{0pt}

\edef\@IAENGeqnarrayTHEmasterstrutheight{0pt} % default height and depth of
\edef\@IAENGeqnarrayTHEmasterstrutdepth{0pt}  % struts within an IAENGeqnarray

\edef\@IAENGeqnarrayTHEmasterstrutHSAVE{0pt} % saved master strut height
\edef\@IAENGeqnarrayTHEmasterstrutDSAVE{0pt} % and depth

\newif\if@IAENGeqnarrayusemasterstrut % flag to indicate that the master strut value
\@IAENGeqnarrayusemasterstruttrue     % is to be used



% saves the strut height and depth of the master strut
\def\@IAENGeqnarraymasterstrutsave{\relax%
\expandafter\skip0=\@IAENGeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEmasterstrutdepth\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% save values
\edef\@IAENGeqnarrayTHEmasterstrutHSAVE{\the\dimen0}%
\edef\@IAENGeqnarrayTHEmasterstrutDSAVE{\the\dimen2}}

% restores the strut height and depth of the master strut
\def\@IAENGeqnarraymasterstrutrestore{\relax%
\expandafter\skip0=\@IAENGeqnarrayTHEmasterstrutHSAVE\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEmasterstrutDSAVE\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% restore values
\edef\@IAENGeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEmasterstrutdepth{\the\dimen2}}


% globally restores the strut height and depth to the 
% master values and sets the master strut flag to true
\def\@IAENGeqnarraystrutreset{\relax%
\expandafter\skip0=\@IAENGeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEmasterstrutdepth\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% restore values
\xdef\@IAENGeqnarrayTHEstrutheight{\the\dimen0}%
\xdef\@IAENGeqnarrayTHEstrutdepth{\the\dimen2}%
\global\@IAENGeqnarrayusemasterstruttrue}


% if the master strut is not to be used, make the current
% values of \@IAENGeqnarrayTHEstrutheight, \@IAENGeqnarrayTHEstrutdepth
% and the use master strut flag, global
% this allows user strut commands issued in the last column to be carried
% into the isolation/strut column
\def\@IAENGeqnarrayglobalizestrutstatus{\relax%
\if@IAENGeqnarrayusemasterstrut\else%
\xdef\@IAENGeqnarrayTHEstrutheight{\@IAENGeqnarrayTHEstrutheight}%
\xdef\@IAENGeqnarrayTHEstrutdepth{\@IAENGeqnarrayTHEstrutdepth}%
\global\@IAENGeqnarrayusemasterstrutfalse%
\fi}



% usage: \IAENGeqnarraystrutsize{height}{depth}[font size commands]
% If called outside the lines of an IAENGeqnarray, sets the height
% and depth of both the master and local struts. If called inside
% an IAENGeqnarray line, sets the height and depth of the local strut
% only and sets the flag to indicate the use of the local strut
% values. If the height or depth is left blank, 0.7\normalbaselineskip
% and 0.3\normalbaselineskip will be used, respectively.
% The optional argument can be used to evaluate the lengths under
% a different font size and styles. If none is specified, the current
% font is used.
% uses scratch registers \skip0, \skip2, \skip3, \dimen0, \dimen2
\def\IAENGeqnarraystrutsize#1#2{\relax\@ifnextchar[{\@IAENGeqnarraystrutsize{#1}{#2}}{\@IAENGeqnarraystrutsize{#1}{#2}[\relax]}}
\def\@IAENGeqnarraystrutsize#1#2[#3]{\def\@IAENGeqnarraystrutsizeARG{#1}%
\ifx\@IAENGeqnarraystrutsizeARG\@empty%
{\setbox0=\hbox{#3\relax\global\skip3=0.7\normalbaselineskip}}%
\skip0=\skip3\relax%
\else% arg one present
{\setbox0=\hbox{#3\relax\global\skip3=#1\relax}}%
\skip0=\skip3\relax%
\fi% if null arg
\def\@IAENGeqnarraystrutsizeARG{#2}%
\ifx\@IAENGeqnarraystrutsizeARG\@empty%
{\setbox0=\hbox{#3\relax\global\skip3=0.3\normalbaselineskip}}%
\skip2=\skip3\relax%
\else% arg two present
{\setbox0=\hbox{#3\relax\global\skip3=#2\relax}}%
\skip2=\skip3\relax%
\fi% if null arg
% remove stretchability, just to be safe
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
\if@IAENGeqnarrayISinner% inner does not touch master strut size
\edef\@IAENGeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEstrutdepth{\the\dimen2}%
\@IAENGeqnarrayusemasterstrutfalse% do not use master
\else% outer, have to set master strut too
\edef\@IAENGeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEmasterstrutdepth{\the\dimen2}%
\edef\@IAENGeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEstrutdepth{\the\dimen2}%
\@IAENGeqnarrayusemasterstruttrue% use master strut
\fi}


% usage: \IAENGeqnarraystrutsizeadd{added height}{added depth}[font size commands]
% If called outside the lines of an IAENGeqnarray, adds the given height
% and depth to both the master and local struts.
% If called inside an IAENGeqnarray line, adds the given height and depth
% to the local strut only and sets the flag to indicate the use 
% of the local strut values.
% In both cases, if a height or depth is left blank, 0pt is used instead.
% The optional argument can be used to evaluate the lengths under
% a different font size and styles. If none is specified, the current
% font is used.
% uses scratch registers \skip0, \skip2, \skip3, \dimen0, \dimen2
\def\IAENGeqnarraystrutsizeadd#1#2{\relax\@ifnextchar[{\@IAENGeqnarraystrutsizeadd{#1}{#2}}{\@IAENGeqnarraystrutsizeadd{#1}{#2}[\relax]}}
\def\@IAENGeqnarraystrutsizeadd#1#2[#3]{\def\@IAENGeqnarraystrutsizearg{#1}%
\ifx\@IAENGeqnarraystrutsizearg\@empty%
\skip0=0pt\relax%
\else% arg one present
{\setbox0=\hbox{#3\relax\global\skip3=#1}}%
\skip0=\skip3\relax%
\fi% if null arg
\def\@IAENGeqnarraystrutsizearg{#2}%
\ifx\@IAENGeqnarraystrutsizearg\@empty%
\skip2=0pt\relax%
\else% arg two present
{\setbox0=\hbox{#3\relax\global\skip3=#2}}%
\skip2=\skip3\relax%
\fi% if null arg
% remove stretchability, just to be safe
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
\if@IAENGeqnarrayISinner% inner does not touch master strut size
% get local strut size
\expandafter\skip0=\@IAENGeqnarrayTHEstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEstrutdepth\relax%
% add it to the user supplied values
\advance\dimen0 by \skip0\relax%
\advance\dimen2 by \skip2\relax%
% update the local strut size
\edef\@IAENGeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEstrutdepth{\the\dimen2}%
\@IAENGeqnarrayusemasterstrutfalse% do not use master
\else% outer, have to set master strut too
% get master strut size
\expandafter\skip0=\@IAENGeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEmasterstrutdepth\relax%
% add it to the user supplied values
\advance\dimen0 by \skip0\relax%
\advance\dimen2 by \skip2\relax%
% update the local and master strut sizes
\edef\@IAENGeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEmasterstrutdepth{\the\dimen2}%
\edef\@IAENGeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IAENGeqnarrayTHEstrutdepth{\the\dimen2}%
\@IAENGeqnarrayusemasterstruttrue% use master strut
\fi}


% allow user a way to see the struts
\newif\ifIAENGvisiblestruts
\IAENGvisiblestrutsfalse

% inserts an invisible strut using the master or local strut values
% uses scratch registers \skip0, \skip2, \dimen0, \dimen2
\def\@IAENGeqnarrayinsertstrut{\relax%
\if@IAENGeqnarrayusemasterstrut
% get master strut size
\expandafter\skip0=\@IAENGeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEmasterstrutdepth\relax%
\else%
% get local strut size
\expandafter\skip0=\@IAENGeqnarrayTHEstrutheight\relax%
\expandafter\skip2=\@IAENGeqnarrayTHEstrutdepth\relax%
\fi%
% remove stretchability, probably not needed
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
% allow user to see struts if desired
\ifIAENGvisiblestruts%
\vrule width0.2pt height\dimen0 depth\dimen2\relax%
\else%
\vrule width0pt height\dimen0 depth\dimen2\relax\fi}


% creates an invisible strut, useable even outside \IAENGeqnarray
% if \IAENGvisiblestrutstrue, the strut will be visible and 0.2pt wide. 
% usage: \IAENGstrut[height][depth][font size commands]
% default is \IAENGstrut[0.7\normalbaselineskip][0.3\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \dimen0, \dimen2, \skip0, \skip2
\def\IAENGstrut{\relax\@ifnextchar[{\@IAENGstrut}{\@IAENGstrut[0.7\normalbaselineskip]}}
\def\@IAENGstrut[#1]{\relax\@ifnextchar[{\@@IAENGstrut[#1]}{\@@IAENGstrut[#1][0.3\normalbaselineskip]}}
\def\@@IAENGstrut[#1][#2]{\relax\@ifnextchar[{\@@@IAENGstrut[#1][#2]}{\@@@IAENGstrut[#1][#2][\relax]}}
\def\@@@IAENGstrut[#1][#2][#3]{\mbox{#3\relax%
\def\@IAENGstrutARG{#1}%
\ifx\@IAENGstrutARG\@empty%
\skip0=0.7\normalbaselineskip\relax%
\else%
\skip0=#1\relax%
\fi%
\def\@IAENGstrutARG{#2}%
\ifx\@IAENGstrutARG\@empty%
\skip2=0.3\normalbaselineskip\relax%
\else%
\skip2=#2\relax%
\fi%
% remove stretchability, probably not needed
\dimen0\skip0\relax%
\dimen2\skip2\relax%
\ifIAENGvisiblestruts%
\vrule width0.2pt height\dimen0 depth\dimen2\relax%
\else%
\vrule width0.0pt height\dimen0 depth\dimen2\relax\fi}}


% enables strut mode by setting a default strut size and then zeroing the
% \baselineskip, \lineskip, \lineskiplimit and \jot
\def\IAENGeqnarraystrutmode{\IAENGeqnarraystrutsize{0.7\normalbaselineskip}{0.3\normalbaselineskip}[\relax]%
\baselineskip=0pt\lineskip=0pt\lineskiplimit=0pt\jot=0pt}



\def\IAENGeqnarray{\@IAENGeqnarraystarformfalse\@IAENGeqnarray}
\def\endIAENGeqnarray{\end@IAENGeqnarray}

\@namedef{IAENGeqnarray*}{\@IAENGeqnarraystarformtrue\@IAENGeqnarray}
\@namedef{endIAENGeqnarray*}{\end@IAENGeqnarray}


% \IAENGeqnarray is an enhanced \eqnarray. 
% The star form defaults to not putting equation numbers at the end of each row.
% usage: \IAENGeqnarray[decl]{cols}
\def\@IAENGeqnarray{\relax\@ifnextchar[{\@@IAENGeqnarray}{\@@IAENGeqnarray[\relax]}}
\def\@@IAENGeqnarray[#1]#2{%
   % default to showing the equation number or not based on whether or not
   % the star form was involked
   \if@IAENGeqnarraystarform\global\@eqnswfalse
   \else% not the star form
   \global\@eqnswtrue
   \fi% if star form
   \@IAENGissubequationfalse% default to no subequations
   \@IAENGlastlinewassubequationfalse% assume last line is not a sub equation
   \@IAENGeqnarrayISinnerfalse% not yet within the lines of the halign
   \@IAENGeqnarraystrutsize{0pt}{0pt}[\relax]% turn off struts by default
   \@IAENGeqnarrayusemasterstruttrue% use master strut till user asks otherwise
   \IAENGvisiblestrutsfalse% diagnostic mode defaults to off
   % no extra space unless the user specifically requests it
   \lineskip=0pt\relax
   \lineskiplimit=0pt\relax
   \baselineskip=\normalbaselineskip\relax%
   \jot=\IAENGnormaljot\relax%
   \mathsurround\z@\relax% no extra spacing around math
   \@advanceIAENGeqncolcnttrue% advance the col counter for each col the user uses, 
                             % used in \IAENGeqnarraymulticol and in the preamble build
   \stepcounter{equation}% advance equation counter before first line
   \setcounter{IAENGsubequation}{0}% no subequation yet 
   \def\@currentlabel{\p@equation\theequation}% redefine the ref label
   \IAENGeqnarraydecl\relax% allow a way for the user to make global overrides
   #1\relax% allow user to override defaults
   \let\\\@IAENGeqnarraycr% replace newline with one that can put in eqn. numbers
   \global\@IAENGeqncolcnt\z@% col. count = 0 for first line
   \@IAENGbuildpreamble #2\end\relax% build the preamble and put it into \@IAENGtrantmptoksA 
   % put in the column for the equation number
   \ifnum\@IAENGeqnnumcols>0\relax\@IAENGappendtoksA{&}\fi% col separator for those after the first
   \toks0={##}%
   % advance the \@IAENGeqncolcnt for the isolation col, this helps with error checking
   \@IAENGappendtoksA{\global\advance\@IAENGeqncolcnt by 1\relax}%
   % add the isolation column
   \@IAENGappendtoksA{\tabskip\z@skip\bgroup\the\toks0\egroup}%
   % advance the \@IAENGeqncolcnt for the equation number col, this helps with error checking
   \@IAENGappendtoksA{&\global\advance\@IAENGeqncolcnt by 1\relax}%
   % add the equation number col to the preamble
   \@IAENGappendtoksA{\tabskip\z@skip\hb@xt@\z@\bgroup\hss\the\toks0\egroup}%
   % note \@IAENGeqnnumcols does not count the equation col or isolation col
   % set the starting tabskip glue as determined by the preamble build
   \tabskip=\@IAENGBPstartglue\relax
   % begin the display alignment
   \@IAENGeqnarrayISinnertrue% commands are now within the lines
   $$\everycr{}\halign to\displaywidth\bgroup
   % "exspand" the preamble
   \span\the\@IAENGtrantmptoksA\cr}

% enter isolation/strut column (or the next column if the user did not use
% every column), record the strut status, complete the columns, do the strut if needed,
% restore counters to correct values and exit
\def\end@IAENGeqnarray{\@IAENGeqnarrayglobalizestrutstatus&\@@IAENGeqnarraycr\egroup%
\if@IAENGlastlinewassubequation\global\advance\c@IAENGsubequation\m@ne\fi%
\global\advance\c@equation\m@ne%
$$\@ignoretrue}

% need a way to remember if last line is a subequation
\newif\if@IAENGlastlinewassubequation%
\@IAENGlastlinewassubequationfalse

% IAENGeqnarray uses a modifed \\ instead of the plain \cr to
% end rows. This allows for things like \\*[vskip amount]
% This "cr" macros are modified versions those for LaTeX2e's eqnarray
% the {\ifnum0=`} braces must be kept away from the last column to avoid
% altering spacing of its math, so we use & to advance to the next column
% as there is an isolation/strut column after the user's columns
\def\@IAENGeqnarraycr{\@IAENGeqnarrayglobalizestrutstatus&% save strut status and advance to next column
   {\ifnum0=`}\fi
   \@ifstar{%
      \global\@eqpen\@M\@IAENGeqnarrayYCR
   }{%
      \global\@eqpen\interdisplaylinepenalty \@IAENGeqnarrayYCR
   }%
}

\def\@IAENGeqnarrayYCR{\@testopt\@IAENGeqnarrayXCR\z@skip}

\def\@IAENGeqnarrayXCR[#1]{%
   \ifnum0=`{\fi}%
   \@@IAENGeqnarraycr
   \noalign{\penalty\@eqpen\vskip\jot\vskip #1\relax}}%

\def\@@IAENGeqnarraycr{\@IAENGtrantmptoksA={}% clear token register
    \advance\@IAENGeqncolcnt by -1\relax% adjust col count because of the isolation column
    \ifnum\@IAENGeqncolcnt>\@IAENGeqnnumcols\relax
    \@IAENGclspkgerror{Too many columns within the IAENGeqnarray\MessageBreak
                          environment}%
    {Use fewer \string &'s or put more columns in the IAENGeqnarry column\MessageBreak 
     specifications.}\relax%
    \else
    \loop% add cols if the user did not use them all
    \ifnum\@IAENGeqncolcnt<\@IAENGeqnnumcols\relax
    \@IAENGappendtoksA{&}%
    \advance\@IAENGeqncolcnt by 1\relax% update the col count
    \repeat
    % this number of &'s will take us the the isolation column
    \fi
    % execute the &'s
    \the\@IAENGtrantmptoksA%
    % handle the strut/isolation column
    \@IAENGeqnarrayinsertstrut% do the strut if needed
    \@IAENGeqnarraystrutreset% reset the strut system for next line or IAENGeqnarray
    &% and enter the equation number column
    % is this line needs an equation number, display it and advance the
    % (sub)equation counters, record what type this line was
    \if@eqnsw%
     \if@IAENGissubequation\theIAENGsubequationdis\addtocounter{equation}{1}\stepcounter{IAENGsubequation}%
     \global\@IAENGlastlinewassubequationtrue%
     \else% display a standard equation number, initialize the IAENGsubequation counter
     \theequationdis\stepcounter{equation}\setcounter{IAENGsubequation}{0}%
     \global\@IAENGlastlinewassubequationfalse\fi%
    \fi%
    % reset the eqnsw flag to indicate default preference of the display of equation numbers
    \if@IAENGeqnarraystarform\global\@eqnswfalse\else\global\@eqnswtrue\fi
    \global\@IAENGissubequationfalse% reset the subequation flag
    % reset the number of columns the user actually used
    \global\@IAENGeqncolcnt\z@\relax
    % the real end of the line
    \cr}





% \IAENGeqnarraybox is like \IAENGeqnarray except the box form puts everything
% inside a vtop, vbox, or vcenter box depending on the letter in the second
% optional argument (t,b,c). Vbox is the default. Unlike \IAENGeqnarray,
% equation numbers are not displayed and \IAENGeqnarraybox can be nested.
% \IAENGeqnarrayboxm is for math mode (like \array) and does not put the vbox
% within an hbox.
% \IAENGeqnarrayboxt is for text mode (like \tabular) and puts the vbox within
% a \hbox{$ $} construct.
% \IAENGeqnarraybox will auto detect whether to use \IAENGeqnarrayboxm or 
% \IAENGeqnarrayboxt depending on the math mode.
% The third optional argument specifies the width this box is to be set to -
% natural width is the default.
% The * forms do not add \jot line spacing
% usage: \IAENGeqnarraybox[decl][pos][width]{cols}
\def\IAENGeqnarrayboxm{\@IAENGeqnarraystarformfalse\@IAENGeqnarrayboxHBOXSWfalse\@IAENGeqnarraybox}
\def\endIAENGeqnarrayboxm{\end@IAENGeqnarraybox}
\@namedef{IAENGeqnarrayboxm*}{\@IAENGeqnarraystarformtrue\@IAENGeqnarrayboxHBOXSWfalse\@IAENGeqnarraybox}
\@namedef{endIAENGeqnarrayboxm*}{\end@IAENGeqnarraybox}

\def\IAENGeqnarrayboxt{\@IAENGeqnarraystarformfalse\@IAENGeqnarrayboxHBOXSWtrue\@IAENGeqnarraybox}
\def\endIAENGeqnarrayboxt{\end@IAENGeqnarraybox}
\@namedef{IAENGeqnarrayboxt*}{\@IAENGeqnarraystarformtrue\@IAENGeqnarrayboxHBOXSWtrue\@IAENGeqnarraybox}
\@namedef{endIAENGeqnarrayboxt*}{\end@IAENGeqnarraybox}

\def\IAENGeqnarraybox{\@IAENGeqnarraystarformfalse\ifmmode\@IAENGeqnarrayboxHBOXSWfalse\else\@IAENGeqnarrayboxHBOXSWtrue\fi%
\@IAENGeqnarraybox}
\def\endIAENGeqnarraybox{\end@IAENGeqnarraybox}

\@namedef{IAENGeqnarraybox*}{\@IAENGeqnarraystarformtrue\ifmmode\@IAENGeqnarrayboxHBOXSWfalse\else\@IAENGeqnarrayboxHBOXSWtrue\fi%
\@IAENGeqnarraybox}
\@namedef{endIAENGeqnarraybox*}{\end@IAENGeqnarraybox}

% flag to indicate if the \IAENGeqnarraybox needs to put things into an hbox{$ $} 
% for \vcenter in non-math mode
\newif\if@IAENGeqnarrayboxHBOXSW%
\@IAENGeqnarrayboxHBOXSWfalse

\def\@IAENGeqnarraybox{\relax\@ifnextchar[{\@@IAENGeqnarraybox}{\@@IAENGeqnarraybox[\relax]}}
\def\@@IAENGeqnarraybox[#1]{\relax\@ifnextchar[{\@@@IAENGeqnarraybox[#1]}{\@@@IAENGeqnarraybox[#1][b]}}
\def\@@@IAENGeqnarraybox[#1][#2]{\relax\@ifnextchar[{\@@@@IAENGeqnarraybox[#1][#2]}{\@@@@IAENGeqnarraybox[#1][#2][\relax]}}

% #1 = decl; #2 = t,b,c; #3 = width, #4 = col specs
\def\@@@@IAENGeqnarraybox[#1][#2][#3]#4{\@IAENGeqnarrayISinnerfalse % not yet within the lines of the halign
   \@IAENGeqnarraymasterstrutsave% save current master strut values
   \@IAENGeqnarraystrutsize{0pt}{0pt}[\relax]% turn off struts by default
   \@IAENGeqnarrayusemasterstruttrue% use master strut till user asks otherwise
   \IAENGvisiblestrutsfalse% diagnostic mode defaults to off
   % no extra space unless the user specifically requests it
   \lineskip=0pt\relax%
   \lineskiplimit=0pt\relax%
   \baselineskip=\normalbaselineskip\relax%
   \jot=\IAENGnormaljot\relax%
   \mathsurround\z@\relax% no extra spacing around math
   % the default end glues are zero for an \IAENGeqnarraybox
   \edef\@IAENGeqnarraycolSEPdefaultstart{\@IAENGeqnarraycolSEPzero}% default start glue
   \edef\@IAENGeqnarraycolSEPdefaultend{\@IAENGeqnarraycolSEPzero}% default end glue
   \edef\@IAENGeqnarraycolSEPdefaultmid{\@IAENGeqnarraycolSEPzero}% default inter-column glue
   \@advanceIAENGeqncolcntfalse% do not advance the col counter for each col the user uses, 
                              % used in \IAENGeqnarraymulticol and in the preamble build
   \IAENGeqnarrayboxdecl\relax% allow a way for the user to make global overrides
   #1\relax% allow user to override defaults
   \let\\\@IAENGeqnarrayboxcr% replace newline with one that allows optional spacing
   \@IAENGbuildpreamble #4\end\relax% build the preamble and put it into \@IAENGtrantmptoksA
   % add an isolation column to the preamble to stop \\'s {} from getting into the last col
   \ifnum\@IAENGeqnnumcols>0\relax\@IAENGappendtoksA{&}\fi% col separator for those after the first
   \toks0={##}%
   % add the isolation column to the preamble
   \@IAENGappendtoksA{\tabskip\z@skip\bgroup\the\toks0\egroup}% 
   % set the starting tabskip glue as determined by the preamble build
   \tabskip=\@IAENGBPstartglue\relax
   % begin the alignment
   \everycr{}%
   % use only the very first token to determine the positioning
   % this stops some problems when the user uses more than one letter,
   % but is probably not worth the effort
   % \noindent is used as a delimiter
   \def\@IAENGgrabfirstoken##1##2\noindent{\let\@IAENGgrabbedfirstoken=##1}%
   \@IAENGgrabfirstoken#2\relax\relax\noindent
   % \@IAENGgrabbedfirstoken has the first token, the rest are discarded
   % if we need to put things into and hbox and go into math mode, do so now
   \if@IAENGeqnarrayboxHBOXSW \leavevmode \hbox \bgroup $\fi%
   % use the appropriate vbox type
   \if\@IAENGgrabbedfirstoken t\relax\vtop\else\if\@IAENGgrabbedfirstoken c\relax%
   \vcenter\else\vbox\fi\fi\bgroup%
   \@IAENGeqnarrayISinnertrue% commands are now within the lines
   \ifx#3\relax\halign\else\halign to #3\relax\fi%
   \bgroup
   % "exspand" the preamble
   \span\the\@IAENGtrantmptoksA\cr}

% carry strut status and enter the isolation/strut column, 
% exit from math mode if needed, and exit
\def\end@IAENGeqnarraybox{\@IAENGeqnarrayglobalizestrutstatus% carry strut status
&% enter isolation/strut column
\@IAENGeqnarrayinsertstrut% do strut if needed
\@IAENGeqnarraymasterstrutrestore% restore the previous master strut values
% reset the strut system for next IAENGeqnarray
% (sets local strut values back to previous master strut values)
\@IAENGeqnarraystrutreset%
% ensure last line, exit from halign, close vbox
\crcr\egroup\egroup%
% exit from math mode and close hbox if needed
\if@IAENGeqnarrayboxHBOXSW $\egroup\fi}



% IAENGeqnarraybox uses a modifed \\ instead of the plain \cr to
% end rows. This allows for things like \\[vskip amount]
% This "cr" macros are modified versions those for LaTeX2e's eqnarray
% For IAENGeqnarraybox, \\* is the same as \\
% the {\ifnum0=`} braces must be kept away from the last column to avoid
% altering spacing of its math, so we use & to advance to the isolation/strut column
% carry strut status into isolation/strut column
\def\@IAENGeqnarrayboxcr{\@IAENGeqnarrayglobalizestrutstatus% carry strut status
&% enter isolation/strut column
\@IAENGeqnarrayinsertstrut% do strut if needed
% reset the strut system for next line or IAENGeqnarray
\@IAENGeqnarraystrutreset%
{\ifnum0=`}\fi%
\@ifstar{\@IAENGeqnarrayboxYCR}{\@IAENGeqnarrayboxYCR}}

% test and setup the optional argument to \\[]
\def\@IAENGeqnarrayboxYCR{\@testopt\@IAENGeqnarrayboxXCR\z@skip}

% IAENGeqnarraybox does not automatically increase line spacing by \jot
\def\@IAENGeqnarrayboxXCR[#1]{\ifnum0=`{\fi}%
\cr\noalign{\if@IAENGeqnarraystarform\else\vskip\jot\fi\vskip#1\relax}}



% starts the halign preamble build
\def\@IAENGbuildpreamble{\@IAENGtrantmptoksA={}% clear token register
\let\@IAENGBPcurtype=u%current column type is not yet known
\let\@IAENGBPprevtype=s%the previous column type was the start
\let\@IAENGBPnexttype=u%next column type is not yet known
% ensure these are valid
\def\@IAENGBPcurglue={0pt plus 0pt minus 0pt}%
\def\@IAENGBPcurcolname{@IAENGdefault}% name of current column definition
% currently acquired numerically referenced glue
% use a name that is easier to remember
\let\@IAENGBPcurnum=\@IAENGtrantmpcountA%
\@IAENGBPcurnum=0%
% tracks number of columns in the preamble
\@IAENGeqnnumcols=0%
% record the default end glues
\edef\@IAENGBPstartglue{\@IAENGeqnarraycolSEPdefaultstart}%
\edef\@IAENGBPendglue{\@IAENGeqnarraycolSEPdefaultend}%
% now parse the user's column specifications
\@@IAENGbuildpreamble}


% parses and builds the halign preamble
\def\@@IAENGbuildpreamble#1#2{\let\@@nextIAENGbuildpreamble=\@@IAENGbuildpreamble%
% use only the very first token to check the end
% \noindent is used as a delimiter as \end can be present here
\def\@IAENGgrabfirstoken##1##2\noindent{\let\@IAENGgrabbedfirstoken=##1}%
\@IAENGgrabfirstoken#1\relax\relax\noindent
\ifx\@IAENGgrabbedfirstoken\end\let\@@nextIAENGbuildpreamble=\@@IAENGfinishpreamble\else%
% identify current and next token type
\@IAENGgetcoltype{#1}{\@IAENGBPcurtype}{1}% current, error on invalid
\@IAENGgetcoltype{#2}{\@IAENGBPnexttype}{0}% next, no error on invalid next
% if curtype is a glue, get the glue def
\if\@IAENGBPcurtype g\@IAENGgetcurglue{#1}{\@IAENGBPcurglue}\fi%
% if curtype is a column, get the column def and set the current column name
\if\@IAENGBPcurtype c\@IAENGgetcurcol{#1}\fi%
% if curtype is a numeral, acquire the user defined glue
\if\@IAENGBPcurtype n\@IAENGprocessNcol{#1}\fi%
% process the acquired glue 
\if\@IAENGBPcurtype g\@IAENGprocessGcol\fi%
% process the acquired col 
\if\@IAENGBPcurtype c\@IAENGprocessCcol\fi%
% ready prevtype for next col spec.
\let\@IAENGBPprevtype=\@IAENGBPcurtype%
% be sure and put back the future token(s) as a group
\fi\@@nextIAENGbuildpreamble{#2}}


% executed just after preamble build is completed
% warn about zero cols, and if prevtype type = u, put in end tabskip glue
\def\@@IAENGfinishpreamble#1{\ifnum\@IAENGeqnnumcols<1\relax
\@IAENGclspkgerror{No column specifiers declared for IAENGeqnarray}%
{At least one column type must be declared for each IAENGeqnarray.}%
\fi%num cols less than 1
%if last type undefined, set default end tabskip glue
\if\@IAENGBPprevtype u\@IAENGappendtoksA{\tabskip=\@IAENGBPendglue}\fi}


% Identify and return the column specifier's type code
\def\@IAENGgetcoltype#1#2#3{%
% use only the very first token to determine the type
% \noindent is used as a delimiter as \end can be present here
\def\@IAENGgrabfirstoken##1##2\noindent{\let\@IAENGgrabbedfirstoken=##1}%
\@IAENGgrabfirstoken#1\relax\relax\noindent
% \@IAENGgrabfirstoken has the first token, the rest are discarded
% n = number
% g = glue (any other char in catagory 12)
% c = letter
% e = \end
% u = undefined
% third argument: 0 = no error message, 1 = error on invalid char
\let#2=u\relax% assume invalid until know otherwise
\ifx\@IAENGgrabbedfirstoken\end\let#2=e\else
\ifcat\@IAENGgrabbedfirstoken\relax\else% screen out control sequences
\if0\@IAENGgrabbedfirstoken\let#2=n\else
\if1\@IAENGgrabbedfirstoken\let#2=n\else
\if2\@IAENGgrabbedfirstoken\let#2=n\else
\if3\@IAENGgrabbedfirstoken\let#2=n\else
\if4\@IAENGgrabbedfirstoken\let#2=n\else
\if5\@IAENGgrabbedfirstoken\let#2=n\else
\if6\@IAENGgrabbedfirstoken\let#2=n\else
\if7\@IAENGgrabbedfirstoken\let#2=n\else
\if8\@IAENGgrabbedfirstoken\let#2=n\else
\if9\@IAENGgrabbedfirstoken\let#2=n\else
\ifcat,\@IAENGgrabbedfirstoken\let#2=g\relax
\else\ifcat a\@IAENGgrabbedfirstoken\let#2=c\relax\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi
\if#2u\relax
\if0\noexpand#3\relax\else\@IAENGclspkgerror{Invalid character in column specifications}%
{Only letters, numerals and certain other symbols are allowed \MessageBreak
as IAENGeqnarray column specifiers.}\fi\fi}


% identify the current letter referenced column
% if invalid, use a default column
\def\@IAENGgetcurcol#1{\expandafter\ifx\csname @IAENGeqnarraycolDEF#1\endcsname\@IAENGeqnarraycolisdefined%
\def\@IAENGBPcurcolname{#1}\else% invalid column name
\@IAENGclspkgerror{Invalid column type "#1" in column specifications.\MessageBreak
Using a default centering column instead}%
{You must define IAENGeqnarray column types before use.}%
\def\@IAENGBPcurcolname{@IAENGdefault}\fi}


% identify and return the predefined (punctuation) glue value
\def\@IAENGgetcurglue#1#2{%
% ! = \! (neg small)  -0.16667em (-3/18 em)
% , = \, (small)       0.16667em ( 3/18 em)
% : = \: (med)         0.22222em ( 4/18 em)
% ; = \; (large)       0.27778em ( 5/18 em)
% ' = \quad            1em
% " = \qquad           2em
% . = 0.5\arraycolsep
% / = \arraycolsep
% ? = 2\arraycolsep
% * = 1fil
% + = \@IAENGeqnarraycolSEPcenter
% - = \@IAENGeqnarraycolSEPzero
% Note that all em values are referenced to the math font (textfont2) fontdimen6
% value for 1em.
% 
% use only the very first token to determine the type
% this prevents errant tokens from getting in the main text
% \noindent is used as a delimiter here
\def\@IAENGgrabfirstoken##1##2\noindent{\let\@IAENGgrabbedfirstoken=##1}%
\@IAENGgrabfirstoken#1\relax\relax\noindent
% get the math font 1em value
% LaTeX2e's NFSS2 does not preload the fonts, but \IAENGeqnarray needs
% to gain access to the math (\textfont2) font's spacing parameters.
% So we create a bogus box here that uses the math font to ensure
% that \textfont2 is loaded and ready. If this is not done,
% the \textfont2 stuff here may not work.
% Thanks to Bernd Raichle for his 1997 post on this topic.
{\setbox0=\hbox{$\displaystyle\relax$}}%
% fontdimen6 has the width of 1em (a quad).
\@IAENGtrantmpdimenA=\fontdimen6\textfont2\relax%
% identify the glue value based on the first token
% we discard anything after the first
\if!\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=-0.16667\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if,\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=0.16667\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if:\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=0.22222\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if;\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=0.27778\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if'\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=1\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if"\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=2\@IAENGtrantmpdimenA\edef#2{\the\@IAENGtrantmpdimenA}\else
\if.\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=0.5\arraycolsep\edef#2{\the\@IAENGtrantmpdimenA}\else
\if/\@IAENGgrabbedfirstoken\edef#2{\the\arraycolsep}\else
\if?\@IAENGgrabbedfirstoken\@IAENGtrantmpdimenA=2\arraycolsep\edef#2{\the\@IAENGtrantmpdimenA}\else
\if *\@IAENGgrabbedfirstoken\edef#2{0pt plus 1fil minus 0pt}\else
\if+\@IAENGgrabbedfirstoken\edef#2{\@IAENGeqnarraycolSEPcenter}\else
\if-\@IAENGgrabbedfirstoken\edef#2{\@IAENGeqnarraycolSEPzero}\else
\edef#2{\@IAENGeqnarraycolSEPzero}%
\@IAENGclspkgerror{Invalid predefined inter-column glue type "#1" in\MessageBreak
column specifications. Using a default value of\MessageBreak
0pt instead}%
{Only !,:;'"./?*+ and - are valid predefined glue types in the\MessageBreak 
IAENGeqnarray column specifications.}\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi}



% process a numerical digit from the column specification
% and look up the corresponding user defined glue value
% can transform current type from n to g or a as the user defined glue is acquired
\def\@IAENGprocessNcol#1{\if\@IAENGBPprevtype g%
\@IAENGclspkgerror{Back-to-back inter-column glue specifiers in column\MessageBreak
specifications. Ignoring consecutive glue specifiers\MessageBreak
after the first}%
{You cannot have two or more glue types next to each other\MessageBreak 
in the IAENGeqnarray column specifications.}%
\let\@IAENGBPcurtype=a% abort this glue, future digits will be discarded
\@IAENGBPcurnum=0\relax%
\else% if we previously aborted a glue
\if\@IAENGBPprevtype a\@IAENGBPcurnum=0\let\@IAENGBPcurtype=a%maintain digit abortion
\else%acquire this number
% save the previous type before the numerical digits started
\if\@IAENGBPprevtype n\else\let\@IAENGBPprevsavedtype=\@IAENGBPprevtype\fi%
\multiply\@IAENGBPcurnum by 10\relax%
\advance\@IAENGBPcurnum by #1\relax% add in number, \relax is needed to stop TeX's number scan
\if\@IAENGBPnexttype n\else%close acquisition
\expandafter\ifx\csname @IAENGeqnarraycolSEPDEF\expandafter\romannumeral\number\@IAENGBPcurnum\endcsname\@IAENGeqnarraycolisdefined%
\edef\@IAENGBPcurglue{\csname @IAENGeqnarraycolSEP\expandafter\romannumeral\number\@IAENGBPcurnum\endcsname}%
\else%user glue not defined
\@IAENGclspkgerror{Invalid user defined inter-column glue type "\number\@IAENGBPcurnum" in\MessageBreak
column specifications. Using a default value of\MessageBreak
0pt instead}%
{You must define all IAENGeqnarray numerical inter-column glue types via\MessageBreak
\string\IAENGeqnarraydefcolsep \space before they are used in column specifications.}%
\edef\@IAENGBPcurglue{\@IAENGeqnarraycolSEPzero}%
\fi% glue defined or not
\let\@IAENGBPcurtype=g% change the type to reflect the acquired glue
\let\@IAENGBPprevtype=\@IAENGBPprevsavedtype% restore the prev type before this number glue
\@IAENGBPcurnum=0\relax%ready for next acquisition
\fi%close acquisition, get glue
\fi%discard or acquire number
\fi%prevtype glue or not
}


% process an acquired glue
% add any acquired column/glue pair to the preamble
\def\@IAENGprocessGcol{\if\@IAENGBPprevtype a\let\@IAENGBPcurtype=a%maintain previous glue abortions
\else
% if this is the start glue, save it, but do nothing else 
% as this is not used in the preamble, but before
\if\@IAENGBPprevtype s\edef\@IAENGBPstartglue{\@IAENGBPcurglue}%
\else%not the start glue
\if\@IAENGBPprevtype g%ignore if back to back glues
\@IAENGclspkgerror{Back-to-back inter-column glue specifiers in column\MessageBreak
specifications. Ignoring consecutive glue specifiers\MessageBreak
after the first}%
{You cannot have two or more glue types next to each other\MessageBreak 
in the IAENGeqnarray column specifications.}%
\let\@IAENGBPcurtype=a% abort this glue
\else% not a back to back glue
\if\@IAENGBPprevtype c\relax% if the previoustype was a col, add column/glue pair to preamble
\ifnum\@IAENGeqnnumcols>0\relax\@IAENGappendtoksA{&}\fi
\toks0={##}%
% make preamble advance col counter if this environment needs this
\if@advanceIAENGeqncolcnt\@IAENGappendtoksA{\global\advance\@IAENGeqncolcnt by 1\relax}\fi
% insert the column defintion into the preamble, being careful not to expand
% the column definition
\@IAENGappendtoksA{\tabskip=\@IAENGBPcurglue}%
\@IAENGappendNOEXPANDtoksA{\begingroup\csname @IAENGeqnarraycolPRE}%
\@IAENGappendtoksA{\@IAENGBPcurcolname}%
\@IAENGappendNOEXPANDtoksA{\endcsname}%
\@IAENGappendtoksA{\the\toks0}%
\@IAENGappendNOEXPANDtoksA{\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IAENGeqnarraycolPOST}%
\@IAENGappendtoksA{\@IAENGBPcurcolname}%
\@IAENGappendNOEXPANDtoksA{\endcsname\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\endgroup}%
\advance\@IAENGeqnnumcols by 1\relax%one more column in the preamble
\else% error: non-start glue with no pending column
\@IAENGclspkgerror{Inter-column glue specifier without a prior column\MessageBreak
type in the column specifications. Ignoring this glue\MessageBreak 
specifier}%
{Except for the first and last positions, glue can be placed only\MessageBreak
between column types.}%
\let\@IAENGBPcurtype=a% abort this glue
\fi% previous was a column
\fi% back-to-back glues
\fi% is start column glue
\fi% prev type not a
}


% process an acquired letter referenced column and, if necessary, add it to the preamble
\def\@IAENGprocessCcol{\if\@IAENGBPnexttype g\else
\if\@IAENGBPnexttype n\else
% we have a column followed by something other than a glue (or numeral glue)
% so we must add this column to the preamble now
\ifnum\@IAENGeqnnumcols>0\relax\@IAENGappendtoksA{&}\fi%col separator for those after the first
\if\@IAENGBPnexttype e\@IAENGappendtoksA{\tabskip=\@IAENGBPendglue\relax}\else%put in end glue
\@IAENGappendtoksA{\tabskip=\@IAENGeqnarraycolSEPdefaultmid\relax}\fi% or default mid glue
\toks0={##}%
% make preamble advance col counter if this environment needs this
\if@advanceIAENGeqncolcnt\@IAENGappendtoksA{\global\advance\@IAENGeqncolcnt by 1\relax}\fi
% insert the column definition into the preamble, being careful not to expand
% the column definition
\@IAENGappendNOEXPANDtoksA{\begingroup\csname @IAENGeqnarraycolPRE}%
\@IAENGappendtoksA{\@IAENGBPcurcolname}%
\@IAENGappendNOEXPANDtoksA{\endcsname}%
\@IAENGappendtoksA{\the\toks0}%
\@IAENGappendNOEXPANDtoksA{\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IAENGeqnarraycolPOST}%
\@IAENGappendtoksA{\@IAENGBPcurcolname}%
\@IAENGappendNOEXPANDtoksA{\endcsname\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\endgroup}%
\advance\@IAENGeqnnumcols by 1\relax%one more column in the preamble
\fi%next type not numeral
\fi%next type not glue
}


%%
%% END OF IAENGeqnarry DEFINITIONS
%%




% set up the running headings, this complex because of all the different
% modes IAENGtran supports
\if@twoside
 \ifCLASSOPTIONtechnote
   \def\ps@headings{%
       \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}
       \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}
       \ifCLASSOPTIONdraftcls
            \ifCLASSOPTIONdraftclsnofoot
               \def\@oddfoot{}\def\@evenfoot{}%
            \else
               \def\@oddfoot{\scriptsize\@date\hfil DRAFT}
               \def\@evenfoot{\scriptsize DRAFT\hfil\@date}
            \fi
       \else
            \def\@oddfoot{}\def\@evenfoot{}
       \fi}
 \else % not a technote
   \def\ps@headings{%
       \ifCLASSOPTIONconference
        \def\@oddhead{}
        \def\@evenhead{}
       \else
        \def\@oddhead{\hbox{}\scriptsize\rightmark \hfil \thepage}
        \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}
       \fi
       \ifCLASSOPTIONdraftcls
            \def\@oddhead{\hbox{}\scriptsize\rightmark \hfil \thepage}
            \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}
            \ifCLASSOPTIONdraftclsnofoot
               \def\@oddfoot{}\def\@evenfoot{}%
            \else
               \def\@oddfoot{\scriptsize\@date\hfil DRAFT}
               \def\@evenfoot{\scriptsize DRAFT\hfil\@date}
            \fi
       \else
            \def\@oddfoot{}\def\@evenfoot{}%
       \fi}
 \fi
\else % single side
\def\ps@headings{%
    \ifCLASSOPTIONconference
     \def\@oddhead{}
     \def\@evenhead{}
    \else
     \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}
     \def\@evenhead{}
    \fi
    \ifCLASSOPTIONdraftcls
          \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}
          \def\@evenhead{}
          \ifCLASSOPTIONdraftclsnofoot
             \def\@oddfoot{}
          \else
             \def\@oddfoot{\scriptsize \@date \hfil DRAFT}
          \fi
    \else
         \def\@oddfoot{}
    \fi
    \def\@evenfoot{}}
\fi


% title page style
\def\ps@IAENGtitlepagestyle{\def\@oddfoot{}\def\@evenfoot{}%
\ifCLASSOPTIONconference
   \def\@oddhead{}%
   \def\@evenhead{}%
\else
   \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}%
   \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}%
\fi
\ifCLASSOPTIONdraftcls
   \def\@oddhead{\hbox{}\scriptsize\leftmark \hfil \thepage}%
   \def\@evenhead{\scriptsize\thepage \hfil \leftmark\hbox{}}%
   \ifCLASSOPTIONdraftclsnofoot\else
      \def\@oddfoot{\scriptsize \@date\hfil DRAFT}%
      \def\@evenfoot{\scriptsize DRAFT\hfil \@date}%
   \fi
\else
   % all non-draft mode footers
   \if@IAENGusingpubid
      % for title pages that are using a pubid
      % do not repeat pubid if using peer review option
      \ifCLASSOPTIONpeerreview
      \else
         \footskip 0pt%
         \ifCLASSOPTIONcompsoc
           \def\@oddfoot{\hss\normalfont\scriptsize\raisebox{-1.5\@IAENGnormalsizeunitybaselineskip}[0ex][0ex]{\@IAENGpubid}\hss}%
           \def\@evenfoot{\hss\normalfont\scriptsize\raisebox{-1.5\@IAENGnormalsizeunitybaselineskip}[0ex][0ex]{\@IAENGpubid}\hss}%
         \else
           \def\@oddfoot{\hss\normalfont\footnotesize\raisebox{1.5ex}[1.5ex]{\@IAENGpubid}\hss}%
           \def\@evenfoot{\hss\normalfont\footnotesize\raisebox{1.5ex}[1.5ex]{\@IAENGpubid}\hss}%
         \fi
      \fi
   \fi
\fi}


% peer review cover page style
\def\ps@IAENGpeerreviewcoverpagestyle{%
\def\@oddhead{}\def\@evenhead{}%
\def\@oddfoot{}\def\@evenfoot{}%
\ifCLASSOPTIONdraftcls
   \ifCLASSOPTIONdraftclsnofoot\else
      \def\@oddfoot{\scriptsize \@date\hfil DRAFT}%
      \def\@evenfoot{\scriptsize DRAFT\hfil \@date}%
   \fi
\else
   % non-draft mode footers
   \if@IAENGusingpubid
      \footskip 0pt%
      \ifCLASSOPTIONcompsoc
        \def\@oddfoot{\hss\normalfont\scriptsize\raisebox{-1.5\@IAENGnormalsizeunitybaselineskip}[0ex][0ex]{\@IAENGpubid}\hss}%
        \def\@evenfoot{\hss\normalfont\scriptsize\raisebox{-1.5\@IAENGnormalsizeunitybaselineskip}[0ex][0ex]{\@IAENGpubid}\hss}%
      \else
        \def\@oddfoot{\hss\normalfont\footnotesize\raisebox{1.5ex}[1.5ex]{\@IAENGpubid}\hss}%
        \def\@evenfoot{\hss\normalfont\footnotesize\raisebox{1.5ex}[1.5ex]{\@IAENGpubid}\hss}%
      \fi
   \fi
\fi}


% start with empty headings
\def\rightmark{}\def\leftmark{}


%% Defines the command for putting the header. \footernote{TEXT} is the same
%% as \markboth{TEXT}{TEXT}. 
%% Note that all the text is forced into uppercase, if you have some text
%% that needs to be in lower case, for instance et. al., then either manually
%% set \leftmark and \rightmark or use \MakeLowercase{et. al.} within the
%% arguments to \markboth.
\def\markboth#1#2{\def\leftmark{\@IAENGcompsoconly{\sffamily}\MakeUppercase{#1}}%
\def\rightmark{\@IAENGcompsoconly{\sffamily}\MakeUppercase{#2}}}
\def\footernote#1{\markboth{#1}{#1}}

\def\today{\ifcase\month\or
    January\or February\or March\or April\or May\or June\or
    July\or August\or September\or October\or November\or December\fi
    \space\number\day, \number\year}




%% CITATION AND BIBLIOGRAPHY COMMANDS
%% 
%% V1.6 no longer supports the older, nonstandard \shortcite and \citename setup stuff
% 
% 
% Modify Latex2e \@citex to separate citations with "], ["
\def\@citex[#1]#2{%
  \let\@citea\@empty
  \@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{], [}%
     \edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
     \if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
     \@ifundefined{b@\@citeb}{\mbox{\reset@font\bfseries ?}%
       \G@refundefinedtrue
       \@latex@warning
         {Citation `\@citeb' on page \thepage \space undefined}}%
       {\hbox{\csname b@\@citeb\endcsname}}}}{#1}}

% V1.6 we create hooks for the optional use of Donald Arseneau's
% cite.sty package. cite.sty is "smart" and will notice that the
% following format controls are already defined and will not
% redefine them. The result will be the proper sorting of the
% citation numbers and auto detection of 3 or more entry "ranges" -
% all in IAENG style:  [1], [2], [5]--[7], [12]
% This also allows for an optional note, i.e., \cite[mynote]{..}.
% If the \cite with note has more than one reference, the note will
% be applied to the last of the listed references. It is generally
% desired that if a note is given, only one reference is listed in
% that \cite.
% Thanks to Mr. Arseneau for providing the required format arguments
% to produce the IAENG style.
\def\citepunct{], [}
\def\citedash{]--[}

% V1.7 default to using same font for urls made by url.sty
\AtBeginDocument{\csname url@samestyle\endcsname}

% V1.6 class files should always provide these
\def\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty


% Provide support for the control entries of IAENGtran.bst V1.00 and later.
% V1.7 optional argument allows for a different aux file to be specified in
% order to handle multiple bibliographies. For example, with multibib.sty:
% \newcites{sec}{Secondary Literature}
% \bstctlcite[@auxoutsec]{BSTcontrolhak}
\def\bstctlcite{\@ifnextchar[{\@bstctlcite}{\@bstctlcite[@auxout]}}
\def\@bstctlcite[#1]#2{\@bsphack
  \@for\@citeb:=#2\do{%
    \edef\@citeb{\expandafter\@firstofone\@citeb}%
    \if@filesw\immediate\write\csname #1\endcsname{\string\citation{\@citeb}}\fi}%
  \@esphack}

% V1.6 provide a way for a user to execute a command just before 
% a given reference number - used to insert a \newpage to balance
% the columns on the last page
\edef\@IAENGtriggerrefnum{0}   % the default of zero means that
                              % the command is not executed
\def\@IAENGtriggercmd{\newpage}

% allow the user to alter the triggered command
\long\def\IAENGtriggercmd#1{\long\def\@IAENGtriggercmd{#1}}

% allow user a way to specify the reference number just before the
% command is executed
\def\IAENGtriggeratref#1{\@IAENGtrantmpcountA=#1%
\edef\@IAENGtriggerrefnum{\the\@IAENGtrantmpcountA}}%

% trigger command at the given reference
\def\@IAENGbibitemprefix{\@IAENGtrantmpcountA=\@IAENGtriggerrefnum\relax%
\advance\@IAENGtrantmpcountA by -1\relax%
\ifnum\c@enumiv=\@IAENGtrantmpcountA\relax\@IAENGtriggercmd\relax\fi}


\def\@biblabel#1{[#1]}

% compsoc journals left align the reference numbers
\@IAENGcompsocnotconfonly{\def\@biblabel#1{[#1]\hfill}}

% controls bib item spacing
\def\IAENGbibitemsep{0pt plus .5pt}

\@IAENGcompsocconfonly{\def\IAENGbibitemsep{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}}


\def\thebibliography#1{\section*{\refname}%
    \addcontentsline{toc}{section}{\refname}%
    % V1.6 add some rubber space here and provide a command trigger
    \footnotesize\@IAENGcompsocconfonly{\small}\vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep\relax
    \itemsep \IAENGbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@IAENGlatexbibitem\bibitem%
    \def\bibitem{\@IAENGbibitemprefix\@IAENGlatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
% originally:
%   \sloppy\clubpenalty4000\widowpenalty4000%
% by adding the \interlinepenalty here, we make it more
% difficult, but not impossible, for LaTeX to break within a reference.
% IAENG almost never breaks a reference (but they do it more often with
% technotes). You may get an underfull vbox warning around the bibliography, 
% but the final result will be much more like what IAENG will publish. 
% MDS 11/2000
\ifCLASSOPTIONtechnote\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty100%
\else\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty500\fi%
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist




% TITLE PAGE COMMANDS
% 
% 
% \IAENGmembership is used to produce the sublargesize italic font used to indicate author 
% IAENG membership. compsoc uses a large size sans slant font
\def\IAENGmembership#1{{\@IAENGnotcompsoconly{\sublargesize}\normalfont\@IAENGcompsoconly{\sffamily}\textit{#1}}}
 

% \IAENGauthorrefmark{} produces a footnote type symbol to indicate author affiliation.
% When given an argument of 1 to 9, \IAENGauthorrefmark{} follows the standard LaTeX footnote
% symbol sequence convention. However, for arguments 10 and above, \IAENGauthorrefmark{} 
% reverts to using lower case roman numerals, so it cannot overflow. Do note that you 
% cannot use \footnotemark[] in place of \IAENGauthorrefmark{} within \author as the footnote
% symbols will have been turned off to prevent \thanks from creating footnote marks.
% \IAENGauthorrefmark{} produces a symbol that appears to LaTeX as having zero vertical
% height - this allows for a more compact line packing, but the user must ensure that
% the interline spacing is large enough to prevent \IAENGauthorrefmark{} from colliding
% with the text above.
% V1.7 make this a robust command
\DeclareRobustCommand*{\IAENGauthorrefmark}[1]{\raisebox{0pt}[0pt][0pt]{\textsuperscript{\footnotesize\ensuremath{\ifcase#1\or *\or \dagger\or \ddagger\or%
    \mathsection\or \mathparagraph\or \|\or **\or \dagger\dagger%
    \or \ddagger\ddagger \else\textsuperscript{\expandafter\romannumeral#1}\fi}}}}


% FONT CONTROLS AND SPACINGS FOR CONFERENCE MODE AUTHOR NAME AND AFFILIATION BLOCKS
% 
% The default font styles for the author name and affiliation blocks (confmode)
\def\@IAENGauthorblockNstyle{\normalfont\@IAENGcompsocnotconfonly{\sffamily}\sublargesize\@IAENGcompsocconfonly{\large}}
\def\@IAENGauthorblockAstyle{\normalfont\@IAENGcompsocnotconfonly{\sffamily}\@IAENGcompsocconfonly{\itshape}\normalsize\@IAENGcompsocconfonly{\large}}
% The default if the user does not use an author block
\def\@IAENGauthordefaulttextstyle{\normalfont\@IAENGcompsocnotconfonly{\sffamily}\sublargesize}

% spacing from title (or special paper notice) to author name blocks (confmode)
% can be negative
\def\@IAENGauthorblockconfadjspace{-0.25em}
% compsoc conferences need more space here
\@IAENGcompsocconfonly{\def\@IAENGauthorblockconfadjspace{0.75\@IAENGnormalsizeunitybaselineskip}}

% spacing between name and affiliation blocks (confmode)
% This can be negative.
% IAENG doesn't want any added spacing here, but I will leave these
% controls in place in case they ever change their mind.
% Personally, I like 0.75ex.
%\def\@IAENGauthorblockNtopspace{0.75ex}
%\def\@IAENGauthorblockAtopspace{0.75ex}
\def\@IAENGauthorblockNtopspace{0.0ex}
\def\@IAENGauthorblockAtopspace{0.0ex}
% baseline spacing within name and affiliation blocks (confmode)
% must be positive, spacings below certain values will make 
% the position of line of text sensitive to the contents of the
% line above it i.e., whether or not the prior line has descenders, 
% subscripts, etc. For this reason it is a good idea to keep
% these above 2.6ex
\def\@IAENGauthorblockNinterlinespace{2.6ex}
\def\@IAENGauthorblockAinterlinespace{2.75ex}

% This tracks the required strut size.
% See the \@IAENGauthorhalign command for the actual default value used.
\def\@IAENGauthorblockXinterlinespace{2.7ex}

% variables to retain font size and style across groups
% values given here have no effect as they will be overwritten later
\gdef\@IAENGSAVESTATEfontsize{10}
\gdef\@IAENGSAVESTATEfontbaselineskip{12}
\gdef\@IAENGSAVESTATEfontencoding{OT1}
\gdef\@IAENGSAVESTATEfontfamily{ptm}
\gdef\@IAENGSAVESTATEfontseries{m}
\gdef\@IAENGSAVESTATEfontshape{n}

% saves the current font attributes
\def\@IAENGcurfontSAVE{\global\let\@IAENGSAVESTATEfontsize\f@size%
\global\let\@IAENGSAVESTATEfontbaselineskip\f@baselineskip%
\global\let\@IAENGSAVESTATEfontencoding\f@encoding%
\global\let\@IAENGSAVESTATEfontfamily\f@family%
\global\let\@IAENGSAVESTATEfontseries\f@series%
\global\let\@IAENGSAVESTATEfontshape\f@shape}

% restores the saved font attributes
\def\@IAENGcurfontRESTORE{\fontsize{\@IAENGSAVESTATEfontsize}{\@IAENGSAVESTATEfontbaselineskip}%
\fontencoding{\@IAENGSAVESTATEfontencoding}%
\fontfamily{\@IAENGSAVESTATEfontfamily}%
\fontseries{\@IAENGSAVESTATEfontseries}%
\fontshape{\@IAENGSAVESTATEfontshape}%
\selectfont}


% variable to indicate if the current block is the first block in the column
\newif\if@IAENGprevauthorblockincol   \@IAENGprevauthorblockincolfalse


% the command places a strut with height and depth = \@IAENGauthorblockXinterlinespace
% we use this technique to have complete manual control over the spacing of the lines
% within the halign environment.
% We set the below baseline portion at 30%, the above
% baseline portion at 70% of the total length.
% Responds to changes in the document's \baselinestretch
\def\@IAENGauthorstrutrule{\@IAENGtrantmpdimenA\@IAENGauthorblockXinterlinespace%
\@IAENGtrantmpdimenA=\baselinestretch\@IAENGtrantmpdimenA%
\rule[-0.3\@IAENGtrantmpdimenA]{0pt}{\@IAENGtrantmpdimenA}}


% blocks to hold the authors' names and affilations. 
% Makes formatting easy for conferences
%
% use real definitions in conference mode
% name block
\def\IAENGauthorblockN#1{\relax\@IAENGauthorblockNstyle% set the default text style
\gdef\@IAENGauthorblockXinterlinespace{0pt}% disable strut for spacer row
% the \expandafter hides the \cr in conditional tex, see the array.sty docs
% for details, probably not needed here as the \cr is in a macro
% do a spacer row if needed
\if@IAENGprevauthorblockincol\expandafter\@IAENGauthorblockNtopspaceline\fi
\global\@IAENGprevauthorblockincoltrue% we now have a block in this column
%restore the correct strut value
\gdef\@IAENGauthorblockXinterlinespace{\@IAENGauthorblockNinterlinespace}%
% input the author names
#1%
% end the row if the user did not already
\crcr}
% spacer row for names
\def\@IAENGauthorblockNtopspaceline{\cr\noalign{\vskip\@IAENGauthorblockNtopspace}}
%
% affiliation block
\def\IAENGauthorblockA#1{\relax\@IAENGauthorblockAstyle% set the default text style
\gdef\@IAENGauthorblockXinterlinespace{0pt}%disable strut for spacer row
% the \expandafter hides the \cr in conditional tex, see the array.sty docs
% for details, probably not needed here as the \cr is in a macro
% do a spacer row if needed
\if@IAENGprevauthorblockincol\expandafter\@IAENGauthorblockAtopspaceline\fi
\global\@IAENGprevauthorblockincoltrue% we now have a block in this column
%restore the correct strut value
\gdef\@IAENGauthorblockXinterlinespace{\@IAENGauthorblockAinterlinespace}%
% input the author affiliations
#1%
% end the row if the user did not already
\crcr}
% spacer row for affiliations
\def\@IAENGauthorblockAtopspaceline{\cr\noalign{\vskip\@IAENGauthorblockAtopspace}}


% allow papers to compile even if author blocks are used in modes other
% than conference or peerreviewca. For such cases, we provide dummy blocks.
\ifCLASSOPTIONconference
\else
   \ifCLASSOPTIONpeerreviewca\else
      % not conference or peerreviewca mode
      \def\IAENGauthorblockN#1{#1}%
      \def\IAENGauthorblockA#1{#1}%
   \fi
\fi



% we provide our own halign so as not to have to depend on tabular
\def\@IAENGauthorhalign{\@IAENGauthordefaulttextstyle% default text style
   \lineskip=0pt\relax% disable line spacing
   \lineskiplimit=0pt\relax%
   \baselineskip=0pt\relax%
   \@IAENGcurfontSAVE% save the current font
   \mathsurround\z@\relax% no extra spacing around math
   \let\\\@IAENGauthorhaligncr% replace newline with halign friendly one
   \tabskip=0pt\relax% no column spacing
   \everycr{}% ensure no problems here
   \@IAENGprevauthorblockincolfalse% no author blocks yet
   \def\@IAENGauthorblockXinterlinespace{2.7ex}% default interline space
   \vtop\bgroup%vtop box
   \halign\bgroup&\relax\hfil\@IAENGcurfontRESTORE\relax ##\relax
   \hfil\@IAENGcurfontSAVE\@IAENGauthorstrutrule\cr}

% ensure last line, exit from halign, close vbox
\def\end@IAENGauthorhalign{\crcr\egroup\egroup}

% handle bogus star form
\def\@IAENGauthorhaligncr{{\ifnum0=`}\fi\@ifstar{\@@IAENGauthorhaligncr}{\@@IAENGauthorhaligncr}}

% test and setup the optional argument to \\[]
\def\@@IAENGauthorhaligncr{\@testopt\@@@IAENGauthorhaligncr\z@skip}

% end the line and do the optional spacer
\def\@@@IAENGauthorhaligncr[#1]{\ifnum0=`{\fi}\cr\noalign{\vskip#1\relax}}



% flag to prevent multiple \and warning messages
\newif\if@IAENGWARNand
\@IAENGWARNandtrue

% if in conference or peerreviewca modes, we support the use of \and as \author is a
% tabular environment, otherwise we warn the user that \and is invalid
% outside of conference or peerreviewca modes.
\def\and{\relax} % provide a bogus \and that we will then override

\renewcommand{\and}[1][\relax]{\if@IAENGWARNand\typeout{** WARNING: \noexpand\and is valid only
                               when in conference or peerreviewca}\typeout{modes (line \the\inputlineno).}\fi\global\@IAENGWARNandfalse}

\ifCLASSOPTIONconference%
\renewcommand{\and}[1][\hfill]{\end{@IAENGauthorhalign}#1\begin{@IAENGauthorhalign}}%
\fi
\ifCLASSOPTIONpeerreviewca
\renewcommand{\and}[1][\hfill]{\end{@IAENGauthorhalign}#1\begin{@IAENGauthorhalign}}%
\fi


% page clearing command
% based on LaTeX2e's \cleardoublepage, but allows different page styles
% for the inserted blank pages
\def\@IAENGcleardoublepage#1{\clearpage\if@twoside\ifodd\c@page\else
\hbox{}\thispagestyle{#1}\newpage\if@twocolumn\hbox{}\thispagestyle{#1}\newpage\fi\fi\fi}


% user command to invoke the title page
\def\maketitle{\par%
  \begingroup%
  \normalfont%
  \def\thefootnote{}%  the \thanks{} mark type is empty
  \def\footnotemark{}% and kill space from \thanks within author
  \let\@makefnmark\relax% V1.7, must *really* kill footnotemark to remove all \textsuperscript spacing as well.
  \footnotesize%       equal spacing between thanks lines
  \footnotesep 0.7\baselineskip%see global setting of \footnotesep for more info
  % V1.7 disable \thanks note indention for compsoc
  \@IAENGcompsoconly{\long\def\@makefntext##1{\parindent 1em\noindent\hbox{\@makefnmark}##1}}%
  \normalsize%
  \ifCLASSOPTIONpeerreview
     \newpage\global\@topnum\z@ \@maketitle\@IAENGstatictitlevskip\@IAENGaftertitletext%
     \thispagestyle{IAENGpeerreviewcoverpagestyle}\@thanks%
  \else
     \if@twocolumn%
        \ifCLASSOPTIONtechnote%
           \newpage\global\@topnum\z@ \@maketitle\@IAENGstatictitlevskip\@IAENGaftertitletext%
        \else
           \twocolumn[\@maketitle\@IAENGdynamictitlevspace\@IAENGaftertitletext]%
        \fi
     \else
        \newpage\global\@topnum\z@ \@maketitle\@IAENGstatictitlevskip\@IAENGaftertitletext%
     \fi
     \thispagestyle{IAENGtitlepagestyle}\@thanks%
  \fi
  % pullup page for pubid if used.
  \if@IAENGusingpubid
     \enlargethispage{-\@IAENGpubidpullup}%
  \fi 
  \endgroup
  \setcounter{footnote}{0}\let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}%
  % v1.6b do not clear these as we will need the title again for peer review papers
  % \gdef\@author{}\gdef\@title{}%
  \let\thanks\relax}



% V1.7 parbox to format \@IAENGcompsoctitleabstractindextext
\long\def\@IAENGcompsoctitleabstractindextextbox#1{\parbox{0.915\textwidth}{#1}}

% formats the Title, authors names, affiliations and special paper notice
% THIS IS A CONTROLLED SPACING COMMAND! Do not allow blank lines or unintentional
% spaces to enter the definition - use % at the end of each line
\def\@maketitle{\newpage
\begin{center}%
\ifCLASSOPTIONtechnote% technotes
   {\bfseries\large\@IAENGcompsoconly{\sffamily}\@title\par}\vskip 1.3em{\lineskip .5em\@IAENGcompsoconly{\sffamily}\@author
   \@IAENGspecialpapernotice\par{\@IAENGcompsoconly{\vskip 1.5em\relax
   \@IAENGcompsoctitleabstractindextextbox{\@IAENGcompsoctitleabstractindextext}\par
   \hfill\@IAENGcompsocdiamondline\hfill\hbox{}\par}}}\relax
\else% not a technote
   \vskip0.2em{\Huge\@IAENGcompsoconly{\sffamily}\@IAENGcompsocconfonly{\normalfont\normalsize\vskip 2\@IAENGnormalsizeunitybaselineskip
   \bfseries\Large}\@title\par}\vskip1.0em\par%
   % V1.6 handle \author differently if in conference mode
   \ifCLASSOPTIONconference%
      {\@IAENGspecialpapernotice\mbox{}\vskip\@IAENGauthorblockconfadjspace%
       \mbox{}\hfill\begin{@IAENGauthorhalign}\@author\end{@IAENGauthorhalign}\hfill\mbox{}\par}\relax
   \else% peerreviewca, peerreview or journal
      \ifCLASSOPTIONpeerreviewca
         % peerreviewca handles author names just like conference mode
         {\@IAENGcompsoconly{\sffamily}\@IAENGspecialpapernotice\mbox{}\vskip\@IAENGauthorblockconfadjspace%
          \mbox{}\hfill\begin{@IAENGauthorhalign}\@author\end{@IAENGauthorhalign}\hfill\mbox{}\par
          {\@IAENGcompsoconly{\vskip 1.5em\relax
           \@IAENGcompsoctitleabstractindextextbox{\@IAENGcompsoctitleabstractindextext}\par\hfill
           \@IAENGcompsocdiamondline\hfill\hbox{}\par}}}\relax
      \else% journal or peerreview
         {\lineskip.5em\@IAENGcompsoconly{\sffamily}\sublargesize\@author\@IAENGspecialpapernotice\par
          {\@IAENGcompsoconly{\vskip 1.5em\relax
           \@IAENGcompsoctitleabstractindextextbox{\@IAENGcompsoctitleabstractindextext}\par\hfill
           \@IAENGcompsocdiamondline\hfill\hbox{}\par}}}\relax
      \fi
   \fi
\fi\end{center}}



% V1.7 Computer Society "diamond line" which follows index terms for nonconference papers
\def\@IAENGcompsocdiamondline{\vrule depth 0pt height 0.5pt width 4cm\hspace{7.5pt}%
\raisebox{-3.5pt}{\fontfamily{pzd}\fontencoding{U}\fontseries{m}\fontshape{n}\fontsize{11}{12}\selectfont\char70}%
\hspace{7.5pt}\vrule depth 0pt height 0.5pt width 4cm\relax}

% V1.7 standard LateX2e \thanks, but with \itshape under compsoc. Also make it a \long\def
% We also need to trigger the one-shot footnote rule
\def\@IAENGtriggeroneshotfootnoterule{\global\@IAENGenableoneshotfootnoteruletrue}


\long\def\thanks#1{\footnotemark
    \protected@xdef\@thanks{\@thanks
        \protect\footnotetext[\the\c@footnote]{\@IAENGcompsoconly{\itshape
        \protect\@IAENGtriggeroneshotfootnoterule\relax}\ignorespaces#1}}}
\let\@thanks\@empty

% V1.7 allow \author to contain \par's. This is needed to allow \thanks to contain \par.
\long\def\author#1{\gdef\@author{#1}}


% in addition to setting up IAENGitemize, we need to remove a baselineskip space above and
% below it because \list's \pars introduce blank lines because of the footnote struts.
\def\@IAENGsetupcompsocitemizelist{\def\labelitemi{$\bullet$}%
\setlength{\IAENGlabelindent}{0pt}\setlength{\parskip}{0pt}%
\setlength{\partopsep}{0pt}\setlength{\topsep}{0.5\baselineskip}\vspace{-1\baselineskip}\relax}


% flag for fake non-compsoc \IAENGcompsocthanksitem - prevents line break on very first item
\newif\if@IAENGbreakcompsocthanksitem \@IAENGbreakcompsocthanksitemfalse

\ifCLASSOPTIONcompsoc
% V1.7 compsoc bullet item \thanks
% also, we need to redefine this to destroy the argument in \@IAENGdynamictitlevspace
\long\def\IAENGcompsocitemizethanks#1{\relax\@IAENGbreakcompsocthanksitemfalse\footnotemark
    \protected@xdef\@thanks{\@thanks
        \protect\footnotetext[\the\c@footnote]{\itshape\protect\@IAENGtriggeroneshotfootnoterule
        {\let\IAENGiedlistdecl\relax\protect\begin{IAENGitemize}[\protect\@IAENGsetupcompsocitemizelist]\ignorespaces#1\relax
        \protect\end{IAENGitemize}}\protect\vspace{-1\baselineskip}}}}
\DeclareRobustCommand*{\IAENGcompsocthanksitem}{\item}
\else
% non-compsoc, allow for dual compilation via rerouting to normal \thanks
\long\def\IAENGcompsocitemizethanks#1{\thanks{#1}}
% redirect to "pseudo-par" \hfil\break\indent after swallowing [] from \IAENGcompsocthanksitem[]
\DeclareRobustCommand{\IAENGcompsocthanksitem}{\@ifnextchar [{\@IAENGthanksswallowoptionalarg}%
{\@IAENGthanksswallowoptionalarg[\relax]}}
% be sure and break only after first item, be sure and ignore spaces after optional argument
\def\@IAENGthanksswallowoptionalarg[#1]{\relax\if@IAENGbreakcompsocthanksitem\hfil\break
\indent\fi\@IAENGbreakcompsocthanksitemtrue\ignorespaces}
\fi


% V1.6b define the \IAENGpeerreviewmaketitle as needed
\ifCLASSOPTIONpeerreview
\def\IAENGpeerreviewmaketitle{\@IAENGcleardoublepage{empty}%
\ifCLASSOPTIONtwocolumn
\twocolumn[\@IAENGpeerreviewmaketitle\@IAENGdynamictitlevspace]
\else
\newpage\@IAENGpeerreviewmaketitle\@IAENGstatictitlevskip
\fi
\thispagestyle{IAENGtitlepagestyle}}
\else
% \IAENGpeerreviewmaketitle does nothing if peer review option has not been selected
\def\IAENGpeerreviewmaketitle{\relax}
\fi

% peerreview formats the repeated title like the title in journal papers.
\def\@IAENGpeerreviewmaketitle{\begin{center}\@IAENGcompsoconly{\sffamily}%
\normalfont\normalsize\vskip0.2em{\Huge\@title\par}\vskip1.0em\par
\end{center}}



% V1.6 
% this is a static rubber spacer between the title/authors and the main text
% used for single column text, or when the title appears in the first column
% of two column text (technotes). 
\def\@IAENGstatictitlevskip{{\normalfont\normalsize
% adjust spacing to next text
% v1.6b handle peer review papers
\ifCLASSOPTIONpeerreview
% for peer review papers, the same value is used for both title pages
% regardless of the other paper modes
   \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip
\else
   \ifCLASSOPTIONconference% conference
      \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip%
   \else%
      \ifCLASSOPTIONtechnote% technote
         \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip%
      \else% journal uses more space
         \vskip 2.5\baselineskip plus 0.75\baselineskip minus 0.375\baselineskip%
      \fi
   \fi
\fi}}


% V1.6
% This is a dynamically determined rigid spacer between the title/authors 
% and the main text. This is used only for single column titles over two 
% column text (most common)
% This is bit tricky because we have to ensure that the textheight of the
% main text is an integer multiple of \baselineskip
% otherwise underfull vbox problems may develop in the second column of the
% text on the titlepage
% The possible use of \IAENGpubid must also be taken into account.
\def\@IAENGdynamictitlevspace{{%
    % we run within a group so that all the macros can be forgotten when we are done
    \long\def\thanks##1{\relax}%don't allow \thanks to run when we evaluate the vbox height
    \long\def\IAENGcompsocitemizethanks##1{\relax}%don't allow \IAENGcompsocitemizethanks to run when we evaluate the vbox height
    \normalfont\normalsize% we declare more descriptive variable names
    \let\@IAENGmaintextheight=\@IAENGtrantmpdimenA%height of the main text columns
    \let\@IAENGINTmaintextheight=\@IAENGtrantmpdimenB%height of the main text columns with integer # lines
    % set the nominal and minimum values for the title spacer
    % the dynamic algorithm will not allow the spacer size to
    % become less than \@IAENGMINtitlevspace - instead it will be
    % lengthened
    % default to journal values
    \def\@IAENGNORMtitlevspace{2.5\baselineskip}%
    \def\@IAENGMINtitlevspace{2\baselineskip}%
    % conferences and technotes need tighter spacing
    \ifCLASSOPTIONconference%conference
     \def\@IAENGNORMtitlevspace{1\baselineskip}%
     \def\@IAENGMINtitlevspace{0.75\baselineskip}%
    \fi
    \ifCLASSOPTIONtechnote%technote
      \def\@IAENGNORMtitlevspace{1\baselineskip}%
      \def\@IAENGMINtitlevspace{0.75\baselineskip}%
    \fi%
    % get the height that the title will take up
    \ifCLASSOPTIONpeerreview
       \settoheight{\@IAENGmaintextheight}{\vbox{\hsize\textwidth \@IAENGpeerreviewmaketitle}}%
    \else
       \settoheight{\@IAENGmaintextheight}{\vbox{\hsize\textwidth \@maketitle}}%
    \fi
    \@IAENGmaintextheight=-\@IAENGmaintextheight% title takes away from maintext, so reverse sign
    % add the height of the page textheight
    \advance\@IAENGmaintextheight by \textheight%
    % correct for title pages using pubid
    \ifCLASSOPTIONpeerreview\else
       % peerreview papers use the pubid on the cover page only.
       % And the cover page uses a static spacer.
       \if@IAENGusingpubid\advance\@IAENGmaintextheight by -\@IAENGpubidpullup\fi
    \fi%
    % subtract off the nominal value of the title bottom spacer
    \advance\@IAENGmaintextheight by -\@IAENGNORMtitlevspace%
    % \topskip takes away some too
    \advance\@IAENGmaintextheight by -\topskip%
    % calculate the column height of the main text for lines
    % now we calculate the main text height as if holding
    % an integer number of \normalsize lines after the first
    % and discard any excess fractional remainder
    % we subtracted the first line, because the first line
    % is placed \topskip into the maintext, not \baselineskip like the
    % rest of the lines.
    \@IAENGINTmaintextheight=\@IAENGmaintextheight%
    \divide\@IAENGINTmaintextheight  by \baselineskip%
    \multiply\@IAENGINTmaintextheight  by \baselineskip%
    % now we calculate how much the title spacer height will
    % have to be reduced from nominal (\@IAENGREDUCEmaintextheight is always
    % a positive value) so that the maintext area will contain an integer
    % number of normal size lines
    % we change variable names here (to avoid confusion) as we no longer
    % need \@IAENGINTmaintextheight and can reuse its dimen register
    \let\@IAENGREDUCEmaintextheight=\@IAENGINTmaintextheight%
    \advance\@IAENGREDUCEmaintextheight by -\@IAENGmaintextheight%
    \advance\@IAENGREDUCEmaintextheight by \baselineskip%
    % this is the calculated height of the spacer
    % we change variable names here (to avoid confusion) as we no longer
    % need \@IAENGmaintextheight and can reuse its dimen register
    \let\@IAENGCOMPENSATElen=\@IAENGmaintextheight%
    \@IAENGCOMPENSATElen=\@IAENGNORMtitlevspace% set the nominal value
    % we go with the reduced length if it is smaller than an increase
    \ifdim\@IAENGREDUCEmaintextheight < 0.5\baselineskip\relax%
     \advance\@IAENGCOMPENSATElen by -\@IAENGREDUCEmaintextheight%
     % if the resulting spacer is too small back out and go with an increase instead
     \ifdim\@IAENGCOMPENSATElen<\@IAENGMINtitlevspace\relax%
      \advance\@IAENGCOMPENSATElen by \baselineskip%
     \fi%
    \else%
     % go with an increase because it is closer to the nominal than a decrease
     \advance\@IAENGCOMPENSATElen by -\@IAENGREDUCEmaintextheight%
     \advance\@IAENGCOMPENSATElen by \baselineskip%
    \fi%
    % set the calculated rigid spacer
    \vspace{\@IAENGCOMPENSATElen}}}



% V1.6
% we allow the user access to the last part of the title area
% useful in emergencies such as when a different spacing is needed
% This text is NOT compensated for in the dynamic sizer.
\let\@IAENGaftertitletext=\relax
\long\def\IAENGaftertitletext#1{\def\@IAENGaftertitletext{#1}}

% V1.7 provide a way for users to enter abstract and keywords
% into the onecolumn title are. This text is compensated for
% in the dynamic sizer.
\let\@IAENGcompsoctitleabstractindextext=\relax
\long\def\IAENGcompsoctitleabstractindextext#1{\def\@IAENGcompsoctitleabstractindextext{#1}}
% V1.7 provide a way for users to get the \@IAENGcompsoctitleabstractindextext if
% not in compsoc journal mode - this way abstract and keywords can be placed
% in their conventional position if not in compsoc mode.
\def\IAENGdisplaynotcompsoctitleabstractindextext{%
\ifCLASSOPTIONcompsoc% display if compsoc conf
\ifCLASSOPTIONconference\@IAENGcompsoctitleabstractindextext\fi
\else% or if not compsoc
\@IAENGcompsoctitleabstractindextext\fi}


% command to allow alteration of baselinestretch, but only if the current
% baselineskip is unity. Used to tweak the compsoc abstract and keywords line spacing.
\def\@IAENGtweakunitybaselinestretch#1{{\def\baselinestretch{1}\selectfont
\global\@tempskipa\baselineskip}\ifnum\@tempskipa=\baselineskip%
\def\baselinestretch{#1}\selectfont\fi\relax}


% abstract and keywords are in \small, except 
% for 9pt docs in which they are in \footnotesize
% Because 9pt docs use an 8pt footnotesize, \small
% becomes a rather awkward 8.5pt
\def\@IAENGabskeysecsize{\small}
\ifx\CLASSOPTIONpt\@IAENGptsizenine
 \def\@IAENGabskeysecsize{\footnotesize}
\fi

% compsoc journals use \footnotesize, compsoc conferences use normalsize
\@IAENGcompsoconly{\def\@IAENGabskeysecsize{\footnotesize}}
\@IAENGcompsocconfonly{\def\@IAENGabskeysecsize{\normalsize}}




% V1.6 have abstract and keywords strip leading spaces, pars and newlines
% so that spacing is more tightly controlled.
\def\abstract{\normalfont
    \if@twocolumn
      \@IAENGabskeysecsize\bfseries\textit{\abstractname}---\relax
    \else
      \begin{center}\vspace{-1.78ex}\@IAENGabskeysecsize\textbf{\abstractname}\end{center}\quotation\@IAENGabskeysecsize
    \fi\@IAENGgobbleleadPARNLSP}
% V1.6 IAENG wants only 1 pica from end of abstract to introduction heading when in 
% conference mode (the heading already has this much above it)
\def\endabstract{\relax\ifCLASSOPTIONconference\vspace{0ex}\else\vspace{1.34ex}\fi\par\if@twocolumn\else\endquotation\fi
    \normalfont\normalsize}

\def\IAENGkeywords{\normalfont
    \if@twocolumn
      \@IAENGabskeysecsize\bfseries\textit{\IAENGkeywordsname}---\relax
    \else
      \begin{center}\@IAENGabskeysecsize\textbf{\IAENGkeywordsname}\end{center}\quotation\@IAENGabskeysecsize
    \fi\@IAENGgobbleleadPARNLSP}
\def\endIAENGkeywords{\relax\ifCLASSOPTIONtechnote\vspace{1.34ex}\else\vspace{0.67ex}\fi
    \par\if@twocolumn\else\endquotation\fi%
    \normalfont\normalsize}

% V1.7 compsoc keywords index terms
\ifCLASSOPTIONcompsoc
  \ifCLASSOPTIONconference% compsoc conference
\def\abstract{\normalfont
      \begin{center}\@IAENGabskeysecsize\textbf{\large\abstractname}\end{center}\vskip 0.5\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip
      \if@twocolumn\else\quotation\fi\itshape\@IAENGabskeysecsize%
      \par\@IAENGgobbleleadPARNLSP}
\def\IAENGkeywords{\normalfont\vskip 1.5\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip
      \begin{center}\@IAENGabskeysecsize\textbf{\large\IAENGkeywordsname}\end{center}\vskip 0.5\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip
      \if@twocolumn\else\quotation\fi\itshape\@IAENGabskeysecsize%
      \par\@IAENGgobbleleadPARNLSP}
  \else% compsoc not conference
\def\abstract{\normalfont\@IAENGtweakunitybaselinestretch{1.15}\sffamily
    \if@twocolumn
      \@IAENGabskeysecsize\noindent\textbf{\abstractname}---\relax
    \else
      \begin{center}\vspace{-1.78ex}\@IAENGabskeysecsize\textbf{\abstractname}\end{center}\quotation\@IAENGabskeysecsize%
    \fi\@IAENGgobbleleadPARNLSP}
\def\IAENGkeywords{\normalfont\@IAENGtweakunitybaselinestretch{1.15}\sffamily
    \if@twocolumn
      \@IAENGabskeysecsize\vskip 0.5\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip\noindent
      \textbf{\IAENGkeywordsname}---\relax
    \else
      \begin{center}\@IAENGabskeysecsize\textbf{\IAENGkeywordsname}\end{center}\quotation\@IAENGabskeysecsize%
    \fi\@IAENGgobbleleadPARNLSP}
  \fi
\fi



% gobbles all leading \, \\ and \par, upon finding first token that
% is not a \ , \\ or a \par, it ceases and returns that token
% 
% used to strip leading \, \\ and \par from the input
% so that such things in the beginning of an environment will not
% affect the formatting of the text
\long\def\@IAENGgobbleleadPARNLSP#1{\let\@IAENGswallowthistoken=0%
\let\@IAENGgobbleleadPARNLSPtoken#1%
\let\@IAENGgobbleleadPARtoken=\par%
\let\@IAENGgobbleleadNLtoken=\\%
\let\@IAENGgobbleleadSPtoken=\ %
\def\@IAENGgobbleleadSPMACRO{\ }%
\ifx\@IAENGgobbleleadPARNLSPtoken\@IAENGgobbleleadPARtoken%
\let\@IAENGswallowthistoken=1%
\fi%
\ifx\@IAENGgobbleleadPARNLSPtoken\@IAENGgobbleleadNLtoken%
\let\@IAENGswallowthistoken=1%
\fi%
\ifx\@IAENGgobbleleadPARNLSPtoken\@IAENGgobbleleadSPtoken%
\let\@IAENGswallowthistoken=1%
\fi%
% a control space will come in as a macro
% when it is the last one on a line
\ifx\@IAENGgobbleleadPARNLSPtoken\@IAENGgobbleleadSPMACRO%
\let\@IAENGswallowthistoken=1%
\fi%
% if we have to swallow this token, do so and taste the next one
% else spit it out and stop gobbling
\ifx\@IAENGswallowthistoken 1\let\@IAENGnextgobbleleadPARNLSP=\@IAENGgobbleleadPARNLSP\else%
\let\@IAENGnextgobbleleadPARNLSP=#1\fi%
\@IAENGnextgobbleleadPARNLSP}%




% TITLING OF SECTIONS
\def\@IAENGsectpunct{:\ \,}  % Punctuation after run-in section heading  (headings which are
                            % part of the paragraphs), need little bit more than a single space
                            % spacing from section number to title
% compsoc conferences use regular period/space punctuation
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference
\def\@IAENGsectpunct{.\ }
\fi\fi


\def\@seccntformat#1{\csname the#1dis\endcsname\hskip 0.5em\relax}

\ifCLASSOPTIONcompsoc
% compsoc journals need extra spacing
\ifCLASSOPTIONconference\else
\def\@seccntformat#1{\csname the#1dis\endcsname\hskip 1em\relax}
\fi\fi

%v1.7 put {} after #6 to allow for some types of user font control
%and use \@@par rather than \par
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
     \let\@svsec\@empty
  \else
     \refstepcounter{#1}%
     % load section label and spacer into \@svsec
     \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi%
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@% tempskipa determines whether is treated as a high
     \begingroup #6{\relax% or low level heading
      \noindent % subsections are NOT indented
       % print top level headings. \@svsec is label, #8 is heading title
       % IAENG does not block indent the section title text, it flows like normal
       {\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\@@par}}%
     \endgroup
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\relax\else
               \protect\numberline{\csname the#1\endcsname}\fi#7}%
  \else % printout low level headings
     % svsechd seems to swallow the trailing space, protect it with \mbox{}
     % got rid of sectionmark stuff
     \def\@svsechd{#6{\hskip #3\relax\@svsec #8\@IAENGsectpunct\mbox{}}%
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\relax\else
               \protect\numberline{\csname the#1\endcsname}\fi#7}}%
  \fi%skip down
  \@xsect{#5}}


% section* handler
%v1.7 put {} after #4 to allow for some types of user font control
%and use \@@par rather than \par
\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
     %\begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M #5\par}\endgroup
     % IAENG does not block indent the section title text, it flows like normal
     \begingroup \noindent #4{\relax{\hskip #1}{\interlinepenalty \@M #5\@@par}}\endgroup
  % svsechd swallows the trailing space, protect it with \mbox{}
  \else \def\@svsechd{#4{\hskip #1\relax #5\@IAENGsectpunct\mbox{}}}\fi
  \@xsect{#3}}


%% SECTION heading spacing and font
%%
% arguments are: #1 - sectiontype name
% (for \@sect)   #2 - section level
%                #3 - section heading indent
%                #4 - top separation (absolute value used, neg indicates not to indent main text)
%                     If negative, make stretch parts negative too!
%                #5 - (absolute value used) positive: bottom separation after heading,
%                      negative: amount to indent main text after heading
%                Both #4 and #5 negative means to indent main text and use negative top separation
%                #6 - font control
% You've got to have \normalfont\normalsize in the font specs below to prevent
% trouble when you do something like:
% \section{Note}{\ttfamily TT-TEXT} is known to ... 
% IAENG sometimes REALLY stretches the area before a section
% heading by up to about 0.5in. However, it may not be a good
% idea to let LaTeX have quite this much rubber.
\ifCLASSOPTIONconference%
% IAENG wants section heading spacing to decrease for conference mode
\def\section{\@startsection{section}{1}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus 1ex minus 0ex}{\normalfont\normalsize\centering\scshape}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\itshape}}%
\else % for journals
\def\section{\@startsection{section}{1}{\z@}{3.0ex plus 1.5ex minus 1.5ex}% V1.6 3.0ex from 3.5ex
{0.7ex plus 1ex minus 0ex}{\normalfont\normalsize\centering\scshape}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{3.5ex plus 1.5ex minus 1.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\itshape}}%
\fi

% for both journals and conferences
% decided to put in a little rubber above the section, might help somebody
\def\subsubsection{\@startsection{subsubsection}{3}{\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize\itshape}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize\itshape}}%


% compsoc
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference
% compsoc conference
\def\section{\@startsection{section}{1}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}{\normalfont\large\bfseries}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}{\normalfont\sublargesize\bfseries}}%
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{0ex}{\normalfont\normalsize\bfseries}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize}}%
\else% compsoc journals
% use negative top separation as compsoc journals do not indent paragraphs after section titles
\def\section{\@startsection{section}{1}{\z@}{-3ex plus -2ex minus -1.5ex}%
{0.7ex plus 1ex minus 0ex}{\normalfont\large\sffamily\bfseries\scshape}}%
% Note that subsection and smaller may not be correct for the Computer Society,
% I have to look up an example.
\def\subsection{\@startsection{subsection}{2}{\z@}{-3.5ex plus -1.5ex minus -1.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\sffamily\bfseries}}%
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-2.5ex plus -1ex minus -1ex}%
{0.5ex plus 0.5ex minus 0ex}{\normalfont\normalsize\sffamily\itshape}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{-0ex plus -0.1ex minus -0.1ex}%
{0ex}{\normalfont\normalsize}}%
\fi\fi




%% ENVIRONMENTS
% "box" symbols at end of proofs
\def\IAENGQEDclosed{\mbox{\rule[0pt]{1.3ex}{1.3ex}}} % for a filled box
% V1.6 some journals use an open box instead that will just fit around a closed one
\def\IAENGQEDopen{{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.2pt}\fbox{\rule[0pt]{0pt}{1.3ex}\rule[0pt]{1.3ex}{0pt}}}}
\ifCLASSOPTIONcompsoc
\def\IAENGQED{\IAENGQEDopen}   % default to open for compsoc
\else
\def\IAENGQED{\IAENGQEDclosed} % otherwise default to closed
\fi

% v1.7 name change to avoid namespace collision with amsthm. Also add support
% for an optional argument.
\def\IAENGproof{\@ifnextchar[{\@IAENGproof}{\@IAENGproof[\IAENGproofname]}}
\def\@IAENGproof[#1]{\par\noindent\hspace{2em}{\itshape #1: }}
\def\endIAENGproof{\hspace*{\fill}~\IAENGQED\par}


%\itemindent is set to \z@ by list, so define new temporary variable
\newdimen\@IAENGtmpitemindent
\def\@begintheorem#1#2{\@IAENGtmpitemindent\itemindent\topsep 0pt\rmfamily\trivlist%
    \item[\hskip \labelsep{\indent\itshape #1\ #2:}]\itemindent\@IAENGtmpitemindent}
\def\@opargbegintheorem#1#2#3{\@IAENGtmpitemindent\itemindent\topsep 0pt\rmfamily \trivlist%
% V1.6 IAENG is back to using () around theorem names which are also in italics
% Thanks to Christian Peel for reporting this.
    \item[\hskip\labelsep{\indent\itshape #1\ #2\ (#3):}]\itemindent\@IAENGtmpitemindent}
% V1.7 remove bogus \unskip that caused equations in theorems to collide with
% lines below.
\def\@endtheorem{\endtrivlist}

% V1.6
% display command for the section the theorem is in - so that \thesection
% is not used as this will be in Roman numerals when we want arabic.
% LaTeX2e uses \def\@thmcounter#1{\noexpand\arabic{#1}} for the theorem number
% (second part) display and \def\@thmcountersep{.} as a separator.
% V1.7 intercept calls to the section counter and reroute to \@IAENGthmcounterinsection
% to allow \appendix(ices} to override as needed.
%
% special handler for sections, allows appendix(ices) to override
\gdef\@IAENGthmcounterinsection#1{\arabic{#1}}
% string macro
\edef\@IAENGstringsection{section}

% redefine the #1#2[#3] form of newtheorem to use a hook to \@IAENGthmcounterinsection
% if section in_counter is used
\def\@xnthm#1#2[#3]{%
  \expandafter\@ifdefinable\csname #1\endcsname
    {\@definecounter{#1}\@newctr{#1}[#3]%
     \edef\@IAENGstringtmp{#3}
     \ifx\@IAENGstringtmp\@IAENGstringsection
     \expandafter\xdef\csname the#1\endcsname{%
     \noexpand\@IAENGthmcounterinsection{#3}\@thmcountersep
          \@thmcounter{#1}}%
     \else
     \expandafter\xdef\csname the#1\endcsname{%
       \expandafter\noexpand\csname the#3\endcsname \@thmcountersep
          \@thmcounter{#1}}%
     \fi
     \global\@namedef{#1}{\@thm{#1}{#2}}%
     \global\@namedef{end#1}{\@endtheorem}}}



%% SET UP THE DEFAULT PAGESTYLE
\ps@headings
\pagenumbering{arabic}

% normally the page counter starts at 1
\setcounter{page}{1}
% however, for peerreview the cover sheet is page 0 or page -1
% (for duplex printing)
\ifCLASSOPTIONpeerreview
   \if@twoside
      \setcounter{page}{-1}
   \else
      \setcounter{page}{0}
   \fi
\fi

% standard book class behavior - let bottom line float up and down as
% needed when single sided
\ifCLASSOPTIONtwoside\else\raggedbottom\fi
% if two column - turn on twocolumn, allow word spacings to stretch more and
% enforce a rigid position for the last lines
\ifCLASSOPTIONtwocolumn
% the peer review option delays invoking twocolumn
   \ifCLASSOPTIONpeerreview\else
      \twocolumn
   \fi
\sloppy 
\flushbottom
\fi




% \APPENDIX and \APPENDICES definitions

% This is the \@ifmtarg command from the LaTeX ifmtarg package
% by Peter Wilson (CUA) and Donald Arseneau
% \@ifmtarg is used to determine if an argument to a command
% is present or not.
% For instance:
% \@ifmtarg{#1}{\typeout{empty}}{\typeout{has something}}
% \@ifmtarg is used with our redefined \section command if
% \appendices is invoked.
% The command \section will behave slightly differently depending
% on whether the user specifies a title: 
% \section{My appendix title}
% or not:
% \section{}
% This way, we can eliminate the blank lines where the title
% would be, and the unneeded : after Appendix in the table of
% contents 
\begingroup
\catcode`\Q=3
\long\gdef\@ifmtarg#1{\@xifmtarg#1QQ\@secondoftwo\@firstoftwo\@nil}
\long\gdef\@xifmtarg#1#2Q#3#4#5\@nil{#4}
\endgroup
% end of \@ifmtarg defs


% V1.7
% command that allows the one time saving of the original definition
% of section to \@IAENGappendixsavesection for \appendix or \appendices 
% we don't save \section here as it may be redefined later by other
% packages (hyperref.sty, etc.)
\def\@IAENGsaveoriginalsectiononce{\let\@IAENGappendixsavesection\section
\let\@IAENGsaveoriginalsectiononce\relax}

% neat trick to grab and process the argument from \section{argument}
% we process differently if the user invoked \section{} with no
% argument (title)
% note we reroute the call to the old \section*
\def\@IAENGprocessthesectionargument#1{%
\@ifmtarg{#1}{%
\@IAENGappendixsavesection*{\appendixname~\thesectiondis}%
\addcontentsline{toc}{section}{\appendixname~\thesection}}{%
\@IAENGappendixsavesection*{\appendixname~\thesectiondis \\* #1}%
\addcontentsline{toc}{section}{\appendixname~\thesection: #1}}}

% we use this if the user calls \section{} after
% \appendix-- which has no meaning. So, we ignore the
% command and its argument. Then, warn the user.
\def\@IAENGdestroythesectionargument#1{\typeout{** WARNING: Ignoring useless
\protect\section\space in Appendix (line \the\inputlineno).}}


% remember \thesection forms will be displayed in \ref calls
% and in the Table of Contents.
% The \sectiondis form is used in the actual heading itself

% appendix command for one single appendix
% normally has no heading. However, if you want a 
% heading, you can do so via the optional argument:
% \appendix[Optional Heading]
\def\appendix{\relax}
\renewcommand{\appendix}[1][]{\@IAENGsaveoriginalsectiononce\par
    % v1.6 keep hyperref's identifiers unique
    \gdef\theHsection{Appendix.A}%
    % v1.6 adjust hyperref's string name for the section
    \xdef\Hy@chapapp{appendix}%
    \setcounter{section}{0}%
    \setcounter{subsection}{0}%
    \setcounter{subsubsection}{0}%
    \setcounter{paragraph}{0}%
    \gdef\thesection{A}%
    \gdef\thesectiondis{}% 
    \gdef\thesubsection{\Alph{subsection}}%
    \gdef\@IAENGthmcounterinsection##1{A}
    \refstepcounter{section}% update the \ref counter
    \@ifmtarg{#1}{\@IAENGappendixsavesection*{\appendixname}%
                  \addcontentsline{toc}{section}{\appendixname}}{%
             \@IAENGappendixsavesection*{\appendixname~\\* #1}%
             \addcontentsline{toc}{section}{\appendixname: #1}}%
    % redefine \section command for appendix
    % leave \section* as is
    \def\section{\@ifstar{\@IAENGappendixsavesection*}{%
                    \@IAENGdestroythesectionargument}}% throw out the argument
                                                     % of the normal form
}



% appendices command for multiple appendices
% user then calls \section with an argument (possibly empty) to
% declare the individual appendices
\def\appendices{\@IAENGsaveoriginalsectiononce\par
    % v1.6 keep hyperref's identifiers unique
    \gdef\theHsection{Appendix.\Alph{section}}%
    % v1.6 adjust hyperref's string name for the section
    \xdef\Hy@chapapp{appendix}%
    \setcounter{section}{-1}% we want \refstepcounter to use section 0
    \setcounter{subsection}{0}%
    \setcounter{subsubsection}{0}%
    \setcounter{paragraph}{0}%
    \ifCLASSOPTIONromanappendices%
    \gdef\thesection{\Roman{section}}%
    \gdef\thesectiondis{\Roman{section}}%
    \@IAENGcompsocconfonly{\gdef\thesectiondis{\Roman{section}.}}%
    \gdef\@IAENGthmcounterinsection##1{A\arabic{##1}}
    \else%
    \gdef\thesection{\Alph{section}}%
    \gdef\thesectiondis{\Alph{section}}%
    \@IAENGcompsocconfonly{\gdef\thesectiondis{\Alph{section}.}}%
    \gdef\@IAENGthmcounterinsection##1{\Alph{##1}}
    \fi%
    \refstepcounter{section}% update the \ref counter
    \setcounter{section}{0}% NEXT \section will be the FIRST appendix
    % redefine \section command for appendices
    % leave \section* as is
    \def\section{\@ifstar{\@IAENGappendixsavesection*}{% process the *-form
                    \refstepcounter{section}% or is a new section so,
                    \@IAENGprocessthesectionargument}}% process the argument 
                                                 % of the normal form
}



% \IAENGPARstart
% Definition for the big two line drop cap letter at the beginning of the
% first paragraph of journal papers. The first argument is the first letter
% of the first word, the second argument is the remaining letters of the
% first word which will be rendered in upper case.
% In V1.6 this has been completely rewritten to:
% 
% 1. no longer have problems when the user begins an environment
%    within the paragraph that uses \IAENGPARstart.
% 2. auto-detect and use the current font family
% 3. revise handling of the space at the end of the first word so that
%    interword glue will now work as normal.
% 4. produce correctly aligned edges for the (two) indented lines.
% 
% We generalize things via control macros - playing with these is fun too.
% 
% V1.7 added more control macros to make it easy for IAENGtrantools.sty users
% to change the font style.
% 
% the number of lines that are indented to clear it
% may need to increase if using decenders
\def\@IAENGPARstartDROPLINES{2}
% minimum number of lines left on a page to allow a \@IAENGPARstart
% Does not take into consideration rubber shrink, so it tends to
% be overly cautious
\def\@IAENGPARstartMINPAGELINES{2}
% V1.7 the height of the drop cap is adjusted to match the height of this text
% in the current font (when \IAENGPARstart is called).
\def\@IAENGPARstartHEIGHTTEXT{T}
% the depth the letter is lowered below the baseline
% the height (and size) of the letter is determined by the sum
% of this value and the height of the \@IAENGPARstartHEIGHTTEXT in the current
% font. It is a good idea to set this value in terms of the baselineskip
% so that it can respond to changes therein.
\def\@IAENGPARstartDROPDEPTH{1.1\baselineskip}
% V1.7 the font the drop cap will be rendered in,
% can take zero or one argument.
\def\@IAENGPARstartFONTSTYLE{\bfseries}
% V1.7 any additional, non-font related commands needed to modify
% the drop cap letter, can take zero or one argument.
\def\@IAENGPARstartCAPSTYLE{\MakeUppercase}
% V1.7 the font that will be used to render the rest of the word,
% can take zero or one argument.
\def\@IAENGPARstartWORDFONTSTYLE{\relax}
% V1.7 any additional, non-font related commands needed to modify
% the rest of the word, can take zero or one argument.
\def\@IAENGPARstartWORDCAPSTYLE{\MakeUppercase}
% This is the horizontal separation distance from the drop letter to the main text.
% Lengths that depend on the font (e.g., ex, em, etc.) will be referenced
% to the font that is active when \IAENGPARstart is called. 
\def\@IAENGPARstartSEP{0.15em}
% V1.7 horizontal offset applied to the left of the drop cap.
\def\@IAENGPARstartHOFFSET{0em}
% V1.7 Italic correction command applied at the end of the drop cap.
\def\@IAENGPARstartITLCORRECT{\/}

% V1.7 compoc uses nonbold drop cap and small caps word style
\ifCLASSOPTIONcompsoc
\def\@IAENGPARstartFONTSTYLE{\mdseries}
\def\@IAENGPARstartWORDFONTSTYLE{\scshape}
\def\@IAENGPARstartWORDCAPSTYLE{\relax}
\fi

% definition of \IAENGPARstart
% THIS IS A CONTROLLED SPACING AREA, DO NOT ALLOW SPACES WITHIN THESE LINES
% 
% The token \@IAENGPARstartfont will be globally defined after the first use
% of \IAENGPARstart and will be a font command which creates the big letter
% The first argument is the first letter of the first word and the second
% argument is the rest of the first word(s).
\def\IAENGPARstart#1#2{\par{%
% if this page does not have enough space, break it and lets start
% on a new one
\@IAENGtranneedspace{\@IAENGPARstartMINPAGELINES\baselineskip}{\relax}%
% V1.7 move this up here in case user uses \textbf for \@IAENGPARstartFONTSTYLE
% which uses command \leavevmode which causes an unwanted \indent to be issued
\noindent
% calculate the desired height of the big letter
% it extends from the top of \@IAENGPARstartHEIGHTTEXT in the current font
% down to \@IAENGPARstartDROPDEPTH below the current baseline
\settoheight{\@IAENGtrantmpdimenA}{\@IAENGPARstartHEIGHTTEXT}%
\addtolength{\@IAENGtrantmpdimenA}{\@IAENGPARstartDROPDEPTH}%
% extract the name of the current font in bold
% and place it in \@IAENGPARstartFONTNAME
\def\@IAENGPARstartGETFIRSTWORD##1 ##2\relax{##1}%
{\@IAENGPARstartFONTSTYLE{\selectfont\edef\@IAENGPARstartFONTNAMESPACE{\fontname\font\space}%
\xdef\@IAENGPARstartFONTNAME{\expandafter\@IAENGPARstartGETFIRSTWORD\@IAENGPARstartFONTNAMESPACE\relax}}}%
% define a font based on this name with a point size equal to the desired
% height of the drop letter
\font\@IAENGPARstartsubfont\@IAENGPARstartFONTNAME\space at \@IAENGtrantmpdimenA\relax%
% save this value as a counter (integer) value (sp points)
\@IAENGtrantmpcountA=\@IAENGtrantmpdimenA%
% now get the height of the actual letter produced by this font size
\settoheight{\@IAENGtrantmpdimenB}{\@IAENGPARstartsubfont\@IAENGPARstartCAPSTYLE{#1}}%
% If something bogus happens like the first argument is empty or the
% current font is strange, do not allow a zero height.
\ifdim\@IAENGtrantmpdimenB=0pt\relax%
\typeout{** WARNING: IAENGPARstart drop letter has zero height! (line \the\inputlineno)}%
\typeout{ Forcing the drop letter font size to 10pt.}%
\@IAENGtrantmpdimenB=10pt%
\fi%
% and store it as a counter
\@IAENGtrantmpcountB=\@IAENGtrantmpdimenB%
% Since a font size doesn't exactly correspond to the height of the capital
% letters in that font, the actual height of the letter, \@IAENGtrantmpcountB,
% will be less than that desired, \@IAENGtrantmpcountA
% we need to raise the font size, \@IAENGtrantmpdimenA 
% by \@IAENGtrantmpcountA / \@IAENGtrantmpcountB
% But, TeX doesn't have floating point division, so we have to use integer
% division. Hence the use of the counters.
% We need to reduce the denominator so that the loss of the remainder will
% have minimal affect on the accuracy of the result
\divide\@IAENGtrantmpcountB by 200%
\divide\@IAENGtrantmpcountA by \@IAENGtrantmpcountB%
% Then reequalize things when we use TeX's ability to multiply by
% floating point values
\@IAENGtrantmpdimenB=0.005\@IAENGtrantmpdimenA%
\multiply\@IAENGtrantmpdimenB by \@IAENGtrantmpcountA%
% \@IAENGPARstartfont is globaly set to the calculated font of the big letter
% We need to carry this out of the local calculation area to to create the
% big letter.
\global\font\@IAENGPARstartfont\@IAENGPARstartFONTNAME\space at \@IAENGtrantmpdimenB%
% Now set \@IAENGtrantmpdimenA to the width of the big letter
% We need to carry this out of the local calculation area to set the
% hanging indent
\settowidth{\global\@IAENGtrantmpdimenA}{\@IAENGPARstartfont
\@IAENGPARstartCAPSTYLE{#1\@IAENGPARstartITLCORRECT}}}%
% end of the isolated calculation environment
% add in the extra clearance we want
\advance\@IAENGtrantmpdimenA by \@IAENGPARstartSEP\relax%
% add in the optional offset
\advance\@IAENGtrantmpdimenA by \@IAENGPARstartHOFFSET\relax%
% V1.7 don't allow negative offsets to produce negative hanging indents
\@IAENGtrantmpdimenB\@IAENGtrantmpdimenA
\ifnum\@IAENGtrantmpdimenB < 0 \@IAENGtrantmpdimenB 0pt\fi
% \@IAENGtrantmpdimenA has the width of the big letter plus the
% separation space and \@IAENGPARstartfont is the font we need to use
% Now, we make the letter and issue the hanging indent command
% The letter is placed in a box of zero width and height so that other
% text won't be displaced by it.
\hangindent\@IAENGtrantmpdimenB\hangafter=-\@IAENGPARstartDROPLINES%
\makebox[0pt][l]{\hspace{-\@IAENGtrantmpdimenA}%
\raisebox{-\@IAENGPARstartDROPDEPTH}[0pt][0pt]{\hspace{\@IAENGPARstartHOFFSET}%
\@IAENGPARstartfont\@IAENGPARstartCAPSTYLE{#1\@IAENGPARstartITLCORRECT}%
\hspace{\@IAENGPARstartSEP}}}%
{\@IAENGPARstartWORDFONTSTYLE{\@IAENGPARstartWORDCAPSTYLE{\selectfont#2}}}}






% determines if the space remaining on a given page is equal to or greater
% than the specified space of argument one
% if not, execute argument two (only if the remaining space is greater than zero)
% and issue a \newpage
% 
% example: \@IAENGtranneedspace{2in}{\vfill}
% 
% Does not take into consideration rubber shrinkage, so it tends to
% be overly cautious
% Based on an example posted by Donald Arseneau
% Note this macro uses \@IAENGtrantmpdimenB internally for calculations,
% so DO NOT PASS \@IAENGtrantmpdimenB to this routine
% if you need a dimen register, import with \@IAENGtrantmpdimenA instead
\def\@IAENGtranneedspace#1#2{\penalty-100\begingroup%shield temp variable
\@IAENGtrantmpdimenB\pagegoal\advance\@IAENGtrantmpdimenB-\pagetotal% space left
\ifdim #1>\@IAENGtrantmpdimenB\relax% not enough space left
\ifdim\@IAENGtrantmpdimenB>\z@\relax #2\fi%
\newpage%
\fi\endgroup}



% IAENGbiography ENVIRONMENT
% Allows user to enter biography leaving place for picture (adapts to font size)
% As of V1.5, a new optional argument allows you to have a real graphic!
% V1.5 and later also fixes the "colliding biographies" which could happen when a 
% biography's text was shorter than the space for the photo.
% MDS 7/2001
% V1.6 prevent multiple biographies from making multiple TOC entries
\newif\if@IAENGbiographyTOCentrynotmade
\global\@IAENGbiographyTOCentrynotmadetrue

% biography counter so hyperref can jump directly to the biographies
% and not just the previous section
\newcounter{IAENGbiography}
\setcounter{IAENGbiography}{0}

% photo area size
\def\@IAENGBIOphotowidth{1.0in}    % width of the biography photo area
\def\@IAENGBIOphotodepth{1.25in}   % depth (height) of the biography photo area
% area cleared for photo
\def\@IAENGBIOhangwidth{1.14in}    % width cleared for the biography photo area
\def\@IAENGBIOhangdepth{1.25in}    % depth cleared for the biography photo area
                                  % actual depth will be a multiple of 
                                  % \baselineskip, rounded up
\def\@IAENGBIOskipN{4\baselineskip}% nominal value of the vskip above the biography

\newenvironment{IAENGbiography}[2][]{\normalfont\@IAENGcompsoconly{\sffamily}\footnotesize%
\unitlength 1in\parskip=0pt\par\parindent 1em\interlinepenalty500%
% we need enough space to support the hanging indent
% the nominal value of the spacer
% and one extra line for good measure
\@IAENGtrantmpdimenA=\@IAENGBIOhangdepth%
\advance\@IAENGtrantmpdimenA by \@IAENGBIOskipN%
\advance\@IAENGtrantmpdimenA by 1\baselineskip%
% if this page does not have enough space, break it and lets start
% with a new one
\@IAENGtranneedspace{\@IAENGtrantmpdimenA}{\relax}%
% nominal spacer can strech, not shrink use 1fil so user can out stretch with \vfill
\vskip \@IAENGBIOskipN plus 1fil minus 0\baselineskip%
% the default box for where the photo goes
\def\@IAENGtempbiographybox{{\setlength{\fboxsep}{0pt}\framebox{%
\begin{minipage}[b][\@IAENGBIOphotodepth][c]{\@IAENGBIOphotowidth}\centering PLACE\\ PHOTO\\ HERE \end{minipage}}}}%
%
% detect if the optional argument was supplied, this requires the
% \@ifmtarg command as defined in the appendix section above
% and if so, override the default box with what they want
\@ifmtarg{#1}{\relax}{\def\@IAENGtempbiographybox{\mbox{\begin{minipage}[b][\@IAENGBIOphotodepth][c]{\@IAENGBIOphotowidth}%
\centering%
#1%
\end{minipage}}}}% end if optional argument supplied
% Make an entry into the table of contents only if we have not done so before
\if@IAENGbiographyTOCentrynotmade%
% link labels to the biography counter so hyperref will jump
% to the biography, not the previous section
\setcounter{IAENGbiography}{-1}%
\refstepcounter{IAENGbiography}%
\addcontentsline{toc}{section}{Biographies}%
\global\@IAENGbiographyTOCentrynotmadefalse%
\fi%
% one more biography
\refstepcounter{IAENGbiography}%
% Make an entry for this name into the table of contents 
\addcontentsline{toc}{subsection}{#2}%
% V1.6 properly handle if a new paragraph should occur while the
% hanging indent is still active. Do this by redefining \par so
% that it will not start a new paragraph. (But it will appear to the
% user as if it did.) Also, strip any leading pars, newlines, or spaces.
\let\@IAENGBIOORGparCMD=\par% save the original \par command
\edef\par{\hfil\break\indent}% the new \par will not be a "real" \par
\settoheight{\@IAENGtrantmpdimenA}{\@IAENGtempbiographybox}% get height of biography box
\@IAENGtrantmpdimenB=\@IAENGBIOhangdepth%
\@IAENGtrantmpcountA=\@IAENGtrantmpdimenB% countA has the hang depth
\divide\@IAENGtrantmpcountA by \baselineskip%  calculates lines needed to produce the hang depth
\advance\@IAENGtrantmpcountA by 1% ensure we overestimate
% set the hanging indent
\hangindent\@IAENGBIOhangwidth%
\hangafter-\@IAENGtrantmpcountA%
% reference the top of the photo area to the top of a capital T
\settoheight{\@IAENGtrantmpdimenB}{\mbox{T}}%
% set the photo box, give it zero width and height so as not to disturb anything
\noindent\makebox[0pt][l]{\hspace{-\@IAENGBIOhangwidth}\raisebox{\@IAENGtrantmpdimenB}[0pt][0pt]{%
\raisebox{-\@IAENGBIOphotodepth}[0pt][0pt]{\@IAENGtempbiographybox}}}%
% now place the author name and begin the bio text
\noindent\textbf{#2\ }\@IAENGgobbleleadPARNLSP}{\relax\let\par=\@IAENGBIOORGparCMD\par%
% 7/2001 V1.5 detect when the biography text is shorter than the photo area
% and pad the unused area - preventing a collision from the next biography entry
% MDS
\ifnum \prevgraf <\@IAENGtrantmpcountA\relax% detect when the biography text is shorter than the photo
    \advance\@IAENGtrantmpcountA by -\prevgraf% calculate how many lines we need to pad
    \advance\@IAENGtrantmpcountA by -1\relax% we compensate for the fact that we indented an extra line
    \@IAENGtrantmpdimenA=\baselineskip% calculate the length of the padding
    \multiply\@IAENGtrantmpdimenA by \@IAENGtrantmpcountA%
    \noindent\rule{0pt}{\@IAENGtrantmpdimenA}% insert an invisible support strut
\fi%
\par\normalfont}



% V1.6
% added biography without a photo environment
\newenvironment{IAENGbiographynophoto}[1]{%
% Make an entry into the table of contents only if we have not done so before
\if@IAENGbiographyTOCentrynotmade%
% link labels to the biography counter so hyperref will jump
% to the biography, not the previous section
\setcounter{IAENGbiography}{-1}%
\refstepcounter{IAENGbiography}%
\addcontentsline{toc}{section}{Biographies}%
\global\@IAENGbiographyTOCentrynotmadefalse%
\fi%
% one more biography
\refstepcounter{IAENGbiography}%
% Make an entry for this name into the table of contents 
\addcontentsline{toc}{subsection}{#1}%
\normalfont\@IAENGcompsoconly{\sffamily}\footnotesize\interlinepenalty500%
\vskip 4\baselineskip plus 1fil minus 0\baselineskip%
\parskip=0pt\par%
\noindent\textbf{#1\ }\@IAENGgobbleleadPARNLSP}{\relax\par\normalfont}


% provide the user with some old font commands
% got this from article.cls
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}


% SPECIAL PAPER NOTICE COMMANDS
% 
% holds the special notice text
\def\@IAENGspecialpapernotice{\relax}
 
% for special papers, like invited papers, the user can do:
% \IAENGspecialpapernotice{(Invited Paper)} before \maketitle
\def\IAENGspecialpapernotice#1{\ifCLASSOPTIONconference%
\def\@IAENGspecialpapernotice{{\sublargesize\textit{#1}\vspace*{1em}}}%
\else%
\def\@IAENGspecialpapernotice{{\\*[1.5ex]\sublargesize\textit{#1}}\vspace*{-2ex}}%
\fi}




% PUBLISHER ID COMMANDS
% to insert a publisher's ID footer
% V1.6 \IAENGpubid has been changed so that the change in page size and style
% occurs in \maketitle. \IAENGpubid must now be issued prior to \maketitle
% use \IAENGpubidadjcol as before - in the second column of the title page
% These changes allow \maketitle to take the reduced page height into
% consideration when dynamically setting the space between the author 
% names and the maintext.
%
% the amount the main text is pulled up to make room for the
% publisher's ID footer
% IAENG uses about 1.3\baselineskip for journals, 
% dynamic title spacing will clean up the fraction
\def\@IAENGpubidpullup{1.3\baselineskip}
\ifCLASSOPTIONtechnote
% for technotes it must be an integer of baselineskip as there can be no
% dynamic title spacing for two column mode technotes (the title is in the
% in first column) and we should maintain an integer number of lines in the
% second column
% There are some examples (such as older issues of "Transactions on
% Information Theory") in which IAENG really pulls the text off the ID for
% technotes - about 0.55in (or 4\baselineskip). We'll use 2\baselineskip
% and call it even.
\def\@IAENGpubidpullup{2\baselineskip}
\fi

% V1.7 compsoc does not use a pullup
\ifCLASSOPTIONcompsoc
\def\@IAENGpubidpullup{0pt}
\fi

% holds the ID text
\def\@IAENGpubid{\relax}

% flag so \maketitle can tell if \IAENGpubid was called
\newif\if@IAENGusingpubid
\global\@IAENGusingpubidfalse
% issue this command in the page to have the ID at the bottom
% V1.6 use before \maketitle
\def\IAENGpubid#1{\def\@IAENGpubid{#1}\global\@IAENGusingpubidtrue}


% command which will pull up (shorten) the column it is executed in
% to make room for the publisher ID. Place in the second column of
% the title page when using \IAENGpubid
% Is smart enough not to do anything when in single column text or
% if the user hasn't called \IAENGpubid
% currently needed in for the second column of a page with the
% publisher ID. If not needed in future releases, please provide this
% command and define it as \relax for backward compatibility
% v1.6b do not allow command to operate if the peer review option has been 
% selected because \IAENGpubidadjcol will not be on the cover page.
% V1.7 do nothing if compsoc
\def\IAENGpubidadjcol{\ifCLASSOPTIONcompsoc\else\ifCLASSOPTIONpeerreview\else
\if@twocolumn\if@IAENGusingpubid\enlargethispage{-\@IAENGpubidpullup}\fi\fi\fi\fi}

% Special thanks to Peter Wilson, Daniel Luecking, and the other
% gurus at comp.text.tex, for helping me to understand how best to
% implement the IAENGpubid command in LaTeX.



%% Lockout some commands under various conditions

% general purpose bit bucket
\newsavebox{\@IAENGtranrubishbin}

% flags to prevent multiple warning messages
\newif\if@IAENGWARNthanks
\newif\if@IAENGWARNIAENGPARstart
\newif\if@IAENGWARNIAENGbiography
\newif\if@IAENGWARNIAENGbiographynophoto
\newif\if@IAENGWARNIAENGpubid
\newif\if@IAENGWARNIAENGpubidadjcol
\newif\if@IAENGWARNIAENGmembership
\newif\if@IAENGWARNIAENGaftertitletext
\@IAENGWARNthankstrue
\@IAENGWARNIAENGPARstarttrue
\@IAENGWARNIAENGbiographytrue
\@IAENGWARNIAENGbiographynophototrue
\@IAENGWARNIAENGpubidtrue
\@IAENGWARNIAENGpubidadjcoltrue
\@IAENGWARNIAENGmembershiptrue
\@IAENGWARNIAENGaftertitletexttrue


%% Lockout some commands when in various modes, but allow them to be restored if needed
%%
% save commands which might be locked out
% so that the user can later restore them if needed
\let\@IAENGSAVECMDthanks\thanks
\let\@IAENGSAVECMDIAENGPARstart\IAENGPARstart
\let\@IAENGSAVECMDIAENGbiography\IAENGbiography
\let\@IAENGSAVECMDendIAENGbiography\endIAENGbiography
\let\@IAENGSAVECMDIAENGbiographynophoto\IAENGbiographynophoto
\let\@IAENGSAVECMDendIAENGbiographynophoto\endIAENGbiographynophoto
\let\@IAENGSAVECMDIAENGpubid\IAENGpubid
\let\@IAENGSAVECMDIAENGpubidadjcol\IAENGpubidadjcol
\let\@IAENGSAVECMDIAENGmembership\IAENGmembership
\let\@IAENGSAVECMDIAENGaftertitletext\IAENGaftertitletext


% disable \IAENGPARstart when in draft mode
% This may have originally been done because the pre-V1.6 drop letter
% algorithm had problems with a non-unity baselinestretch
% At any rate, it seems too formal to have a drop letter in a draft
% paper.
\ifCLASSOPTIONdraftcls
\def\IAENGPARstart#1#2{#1#2\if@IAENGWARNIAENGPARstart\typeout{** ATTENTION: \noexpand\IAENGPARstart
 is disabled in draft mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGPARstartfalse}
\fi
% and for technotes
\ifCLASSOPTIONtechnote
\def\IAENGPARstart#1#2{#1#2\if@IAENGWARNIAENGPARstart\typeout{** WARNING: \noexpand\IAENGPARstart
 is locked out for technotes (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGPARstartfalse}
\fi


% lockout unneeded commands when in conference mode
\ifCLASSOPTIONconference
% when locked out, \thanks, \IAENGbiography, \IAENGbiographynophoto, \IAENGpubid,
% \IAENGmembership and \IAENGaftertitletext will all swallow their given text. 
% \IAENGPARstart will output a normal character instead
% warn the user about these commands only once to prevent the console screen
% from filling up with redundant messages
\def\thanks#1{\if@IAENGWARNthanks\typeout{** WARNING: \noexpand\thanks
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNthanksfalse}
\def\IAENGPARstart#1#2{#1#2\if@IAENGWARNIAENGPARstart\typeout{** WARNING: \noexpand\IAENGPARstart
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGPARstartfalse}


% LaTeX treats environments and commands with optional arguments differently.
% the actual ("internal") command is stored as \\commandname 
% (accessed via \csname\string\commandname\endcsname )
% the "external" command \commandname is a macro with code to determine
% whether or not the optional argument is presented and to provide the 
% default if it is absent. So, in order to save and restore such a command
% we would have to save and restore \\commandname as well. But, if LaTeX
% ever changes the way it names the internal names, the trick would break.
% Instead let us just define a new environment so that the internal
% name can be left undisturbed.
\newenvironment{@IAENGbogusbiography}[2][]{\if@IAENGWARNIAENGbiography\typeout{** WARNING: \noexpand\IAENGbiography
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGbiographyfalse%
\setbox\@IAENGtranrubishbin\vbox\bgroup}{\egroup\relax}
% and make biography point to our bogus biography
\let\IAENGbiography=\@IAENGbogusbiography
\let\endIAENGbiography=\end@IAENGbogusbiography

\renewenvironment{IAENGbiographynophoto}[1]{\if@IAENGWARNIAENGbiographynophoto\typeout{** WARNING: \noexpand\IAENGbiographynophoto
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGbiographynophotofalse%
\setbox\@IAENGtranrubishbin\vbox\bgroup}{\egroup\relax}

\def\IAENGpubid#1{\if@IAENGWARNIAENGpubid\typeout{** WARNING: \noexpand\IAENGpubid 
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGpubidfalse}
\def\IAENGpubidadjcol{\if@IAENGWARNIAENGpubidadjcol\typeout{** WARNING: \noexpand\IAENGpubidadjcol
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGpubidadjcolfalse}
\def\IAENGmembership#1{\if@IAENGWARNIAENGmembership\typeout{** WARNING: \noexpand\IAENGmembership
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGmembershipfalse}
\def\IAENGaftertitletext#1{\if@IAENGWARNIAENGaftertitletext\typeout{** WARNING: \noexpand\IAENGaftertitletext
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IAENGWARNIAENGaftertitletextfalse}
\fi


% provide a way to restore the commands that are locked out
\def\IAENGoverridecommandlockouts{%
\typeout{** ATTENTION: Overriding command lockouts (line \the\inputlineno).}%
\let\thanks\@IAENGSAVECMDthanks%
\let\IAENGPARstart\@IAENGSAVECMDIAENGPARstart%
\let\IAENGbiography\@IAENGSAVECMDIAENGbiography%
\let\endIAENGbiography\@IAENGSAVECMDendIAENGbiography%
\let\IAENGbiographynophoto\@IAENGSAVECMDIAENGbiographynophoto%
\let\endIAENGbiographynophoto\@IAENGSAVECMDendIAENGbiographynophoto%
\let\IAENGpubid\@IAENGSAVECMDIAENGpubid%
\let\IAENGpubidadjcol\@IAENGSAVECMDIAENGpubidadjcol%
\let\IAENGmembership\@IAENGSAVECMDIAENGmembership%
\let\IAENGaftertitletext\@IAENGSAVECMDIAENGaftertitletext}



% need a backslash character for typeout output
{\catcode`\|=0 \catcode`\\=12
|xdef|@IAENGbackslash{\}}


% hook to allow easy disabling of all legacy warnings
\def\@IAENGlegacywarn#1#2{\typeout{** ATTENTION: \@IAENGbackslash #1 is deprecated (line \the\inputlineno).
Use \@IAENGbackslash #2 instead.}}


% provide for legacy commands
\def\authorblockA{\@IAENGlegacywarn{authorblockA}{IAENGauthorblockA}\IAENGauthorblockA}
\def\authorblockN{\@IAENGlegacywarn{authorblockN}{IAENGauthorblockN}\IAENGauthorblockN}
\def\authorrefmark{\@IAENGlegacywarn{authorrefmark}{IAENGauthorrefmark}\IAENGauthorrefmark}
\def\PARstart{\@IAENGlegacywarn{PARstart}{IAENGPARstart}\IAENGPARstart}
\def\pubid{\@IAENGlegacywarn{pubid}{IAENGpubid}\IAENGpubid}
\def\pubidadjcol{\@IAENGlegacywarn{pubidadjcol}{IAENGpubidadjcol}\IAENGpubidadjcol}
\def\QED{\@IAENGlegacywarn{QED}{IAENGQED}\IAENGQED}
\def\QEDclosed{\@IAENGlegacywarn{QEDclosed}{IAENGQEDclosed}\IAENGQEDclosed}
\def\QEDopen{\@IAENGlegacywarn{QEDopen}{IAENGQEDopen}\IAENGQEDopen}
\def\specialpapernotice{\@IAENGlegacywarn{specialpapernotice}{IAENGspecialpapernotice}\IAENGspecialpapernotice}



% provide for legacy environments
\def\biography{\@IAENGlegacywarn{biography}{IAENGbiography}\IAENGbiography}
\def\biographynophoto{\@IAENGlegacywarn{biographynophoto}{IAENGbiographynophoto}\IAENGbiographynophoto}
\def\keywords{\@IAENGlegacywarn{keywords}{IAENGkeywords}\IAENGkeywords}
\def\endbiography{\endIAENGbiography}
\def\endbiographynophoto{\endIAENGbiographynophoto}
\def\endkeywords{\endIAENGkeywords}


% provide for legacy IED commands/lengths when possible
\let\labelindent\IAENGlabelindent
\def\calcleftmargin{\@IAENGlegacywarn{calcleftmargin}{IAENGcalcleftmargin}\IAENGcalcleftmargin}
\def\setlabelwidth{\@IAENGlegacywarn{setlabelwidth}{IAENGsetlabelwidth}\IAENGsetlabelwidth}
\def\usemathlabelsep{\@IAENGlegacywarn{usemathlabelsep}{IAENGusemathlabelsep}\IAENGusemathlabelsep}
\def\iedlabeljustifyc{\@IAENGlegacywarn{iedlabeljustifyc}{IAENGiedlabeljustifyc}\IAENGiedlabeljustifyc}
\def\iedlabeljustifyl{\@IAENGlegacywarn{iedlabeljustifyl}{IAENGiedlabeljustifyl}\IAENGiedlabeljustifyl}
\def\iedlabeljustifyr{\@IAENGlegacywarn{iedlabeljustifyr}{IAENGiedlabeljustifyr}\IAENGiedlabeljustifyr}



% let \proof use the IAENGtran version even after amsthm is loaded
% \proof is now deprecated in favor of \IAENGproof
\AtBeginDocument{\def\proof{\@IAENGlegacywarn{proof}{IAENGproof}\IAENGproof}\def\endproof{\endIAENGproof}}

% V1.7 \overrideIAENGmargins is no longer supported.
\def\overrideIAENGmargins{%
\typeout{** WARNING: \string\overrideIAENGmargins \space no longer supported (line \the\inputlineno).}%
\typeout{** Use the \string\CLASSINPUTinnersidemargin, \string\CLASSINPUToutersidemargin \space controls instead.}}


\endinput

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of IAENGtran.cls  %%%%%%%%%%%%%%%%%%%%%%%%%%%%


